"""Independent invariants for supervision membership, not an experiment matrix."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from core.breakout_quality_runtime import TrainingPairPartition
from filters.breakout_quality.pair_supervision import resolve_training_pair_membership


class PairPartitionSupervisionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def loss(self, margins, targets, dates, membership=None):
        from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss
        kwargs = {} if membership is None else {
            'pair_partition_membership': membership, 'pair_partition_relation': 'within_positive'}
        return _pairwise_logistic_loss(self.torch, margins, targets, dates,
            reduction='full_list_delta_ndcg_weighted', **kwargs)

    def test_full_list_weights_and_masked_batch_denominator_match_independent_formula(self):
        t=self.torch
        m=t.tensor([2., -1., 0.5, 1., 0.1, 1.5, -0.5, 0.8], requires_grad=True)
        y=t.tensor([0., 1., 0.6, 0.3, 0., 1., 0.6, 0.3])
        dates=np.repeat(['2020-01-02','2020-01-03'],4)
        mask=np.array([True,True,True,False,True,True,False,False])
        loss,count=self.loss(m,y,dates,mask)
        numerator=denominator=0.; pairs=0
        a=m.detach().numpy(); b=y.numpy()
        for offset in (0,4):
            mm,yy=a[offset:offset+4],b[offset:offset+4]
            discounts=1/np.log2(np.arange(4)+2)
            by_item=np.empty(4);by_item[np.argsort(-mm,kind='stable')]=discounts
            idcg=np.sum(np.sort(yy)[::-1]*discounts)
            for i in range(4):
                for j in range(i+1,4):
                    if mask[offset+i] and mask[offset+j] and yy[i]!=yy[j]:
                        weight=abs(yy[i]-yy[j])*abs(by_item[i]-by_item[j])/idcg
                        numerator+=weight*np.logaddexp(0.,-np.sign(yy[i]-yy[j])*(mm[i]-mm[j]))
                        denominator+=weight;pairs+=1
        self.assertEqual(count,pairs)
        self.assertAlmostEqual(loss.item(),numerator/denominator,places=6)
        # A candidate sublist would change cross-date weights and is forbidden.
        subloss,_=self.loss(m[t.tensor(mask)],y[t.tensor(mask)],dates[mask])
        self.assertGreater(abs(loss.item()-subloss.item()),1e-4)
        loss.backward()
        np.testing.assert_array_equal(m.grad.detach().numpy()[~mask],0.)

    def test_all_members_preserve_original_loss_exactly(self):
        t=self.torch;m=t.tensor([0.4,0.2,-0.1]);y=t.tensor([0.,0.5,1.]);dates=['2020-01-02']*3
        a,ac=self.loss(m,y,dates);b,bc=self.loss(m,y,dates,[True]*3)
        self.assertTrue(t.equal(a,b));self.assertEqual(ac,bc)

    def test_zero_pair_date_contributes_neither_numerator_nor_denominator(self):
        t=self.torch;m=t.tensor([1.,0.,-1.,2.,0.]);y=t.tensor([0.,0.5,1.,0.,1.])
        all_loss,count=self.loss(m,y,['2020-01-02']*3+['2020-01-03']*2,[True,False,False,True,True])
        one_loss,one_count=self.loss(m[3:],y[3:],['2020-01-03']*2,[True,True])
        self.assertTrue(t.equal(all_loss,one_loss));self.assertEqual((count,one_count),(1,1))
        self.assertEqual(self.loss(m,y,['2020-01-02']*5,[True,False,False,False,False]),(None,0))
        self.assertEqual(self.loss(m,t.zeros(5),['2020-01-02']*5,[True]*5),(None,0))

    @staticmethod
    def bank(future_multiplier=1.):
        class Bank:
            def __len__(self): return len(self._ticker_ids)
        bank=Bank();dates=pd.bdate_range('2018-01-01',periods=380)
        close=np.full(380,100.);close[310:]=120.;close[345:]=145.
        array=np.column_stack([close,close*1.01,close*.99,close,np.ones(380)])
        array[330:,:4]*=future_multiplier
        bank._frame_arrays=(array,);bank._frame_dates=(dates.to_numpy(),)
        bank._source_positions=np.array([299,309,310,311,320]);bank._ticker_ids=np.zeros(5,dtype=int)
        table=pd.DataFrame({'ticker':['TEST']*5,'date':dates.take(bank._source_positions)})
        return bank,table

    def test_membership_uses_price_prefix_and_ignores_future_labels(self):
        a,table=self.bank();b,other=self.bank(3.)
        mask=resolve_training_pair_membership(TrainingPairPartition(),a,table)
        future=resolve_training_pair_membership(TrainingPairPartition(),b,other)
        np.testing.assert_array_equal(mask,[False,False,True,False,False])
        np.testing.assert_array_equal(mask,future)
        changed=table.assign(raw_target=[9,-9,999,0,1],label=[1,1,0,0,1])
        self.assertIs(mask,resolve_training_pair_membership(TrainingPairPartition(),a,changed))
        self.assertFalse(mask.flags.writeable)

    def test_missing_duplicate_or_misaligned_source_fails_closed(self):
        bank,table=self.bank();p=TrainingPairPartition()
        with self.assertRaises(ValueError): resolve_training_pair_membership(p,bank,None)
        with self.assertRaises(ValueError): resolve_training_pair_membership(p,np.zeros((5,2,2)),table)
        with self.assertRaises(ValueError): resolve_training_pair_membership(p,bank,table.assign(date=table.date.iloc[0]))
        with self.assertRaises(ValueError): resolve_training_pair_membership(p,bank,table.assign(date=table.date+pd.Timedelta(days=1)))
        with self.assertRaises(ValueError): TrainingPairPartition(source='unapproved_future_candidates')


if __name__=='__main__':
    unittest.main()
