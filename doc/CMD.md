# 常用指令

python apps/research.py optimizer timing --trials 30 --seeds 1 --timing-fold-workers 6 

## 環境 / 測試

```bash
python requirements/export_requirements_lock.py
python apps/run_bundle.py
python tools/local_regression/run_all.py --only quick_gate
python tools/validate/preflight_env.py
```

## 打包

```bash
python apps/run_bundle.py
python apps/package_zip.py
```

- 一般交付使用`apps/run_bundle.py`，先commit當前snapshot並建立ZIP，再執行formal test；測試FAIL不回滾該commit或刪除ZIP。
- `apps/package_zip.py`直接模式保留給單純snapshot／歷史相容用途，不取代正式整合入口。

