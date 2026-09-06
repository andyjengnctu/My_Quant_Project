"""User-adjustable downloader policy values.

Keep only declarative knobs here. Downloader I/O, retries, exception handling,
path resolution, and provider loading belong to ``services.downloader``.
"""

DOWNLOADER_MIN_VOLUME = 1_000_000
DOWNLOADER_MIN_MARKET_CAP = 10_000_000_000
DOWNLOADER_RESCAN_DAYS = 7
DOWNLOADER_REQUEST_TIMEOUT_SEC = 10
DOWNLOADER_FINMIND_DOWNLOAD_SLEEP_SEC = 0.5

DOWNLOADER_VERBOSE_UNIVERSE_FETCH_ERRORS = False
DOWNLOADER_VERBOSE_LAST_DATE_CHECK_ERRORS = False
DOWNLOADER_VERBOSE_DOWNLOAD_ERRORS = False
