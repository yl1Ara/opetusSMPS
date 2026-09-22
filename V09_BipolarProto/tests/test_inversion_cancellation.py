import threading
import unittest

import pandas as pd

from DMPS_inversion_gui import online_app


class InversionCancellationTests(unittest.TestCase):
    def test_pre_cancelled_calculation_stops_before_processing(self):
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaisesRegex(online_app.InversionCancelled, "stopped by user"):
            online_app.run_inversion_calculation(pd.DataFrame(), cancel_event)

    def test_cancel_check_allows_missing_event(self):
        online_app.check_inversion_cancelled(None)


if __name__ == "__main__":
    unittest.main()
