import unittest
import json

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from DMPS_inversion_gui import online_app


class OnlineInteractionTests(unittest.TestCase):
    def setUp(self):
        self.previous_modal_analysis = online_app.latest_modal_analysis

    def tearDown(self):
        online_app.latest_modal_analysis = self.previous_modal_analysis

    def heatmap_fixture(self):
        sizes = np.geomspace(5.0, 100.0, 100)
        log_sizes = np.log10(sizes)
        first = 100 / (0.1 * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((log_sizes - np.log10(15.0)) / 0.1) ** 2
        )
        second = 1000 / (0.1 * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((log_sizes - np.log10(40.0)) / 0.1) ** 2
        )
        times = pd.DatetimeIndex(["2026-08-01T00:00:00Z"] * 2)
        result = [{
            "kind": "heatmap", "method": "test", "polarity": "positive",
            "x": times, "y": sizes, "Z": np.column_stack([first, second]),
        }]
        figure = go.Figure(go.Heatmap(
            x=times,
            y=sizes,
            z=np.column_stack([first, second]),
            meta={"kind": "inversion_heatmap", "method": "test", "polarity": "positive"},
        ))
        return result, figure

    def test_heatmap_point_number_selects_duplicate_timestamp_column(self):
        result, figure = self.heatmap_fixture()
        analysis = online_app.analyze_heatmap_click(
            {"points": [{
                "curveNumber": 0,
                "pointNumber": [50, 1],
                "x": "2026-08-01T00:00:00Z",
                "y": 40.0,
            }]},
            figure,
            result,
            "1",
            6.5,
            80.0,
        )

        self.assertEqual(analysis["status"], "ok")
        self.assertAlmostEqual(
            analysis["components"][0]["mode_diameter_nm"], 40.0, delta=0.5
        )

    def test_non_heatmap_click_is_ignored(self):
        figure = go.Figure(go.Scatter(x=[1], y=[2]))
        self.assertIsNone(online_app.analyze_heatmap_click(
            {"points": [{"curveNumber": 0, "x": 1, "y": 2}]},
            figure,
            [],
            "1",
            6.5,
            80.0,
        ))

    def test_custom_modal_panes_do_not_change_save_cache(self):
        result, figure = self.heatmap_fixture()
        analysis = online_app.analyze_heatmap_click(
            {"points": [{
                "curveNumber": 0, "pointNumber": [50, 1],
                "x": "2026-08-01T00:00:00Z", "y": 40.0,
            }]},
            figure,
            result,
            "1",
            6.5,
            80.0,
        )
        sentinel = {"source": "personal-session"}
        online_app.latest_modal_analysis = sentinel

        online_app.render_modal_analysis(
            analysis, pn.pane.Plotly(), pn.pane.Markdown()
        )

        self.assertIs(online_app.latest_modal_analysis, sentinel)

    def test_json_conversion_replaces_nonfinite_values(self):
        converted = online_app._json_safe({
            "values": np.array([1.0, np.nan, np.inf]),
            "scalar": np.float64(-np.inf),
        })

        self.assertEqual(converted, {"values": [1.0, None, None], "scalar": None})
        json.dumps(converted, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
