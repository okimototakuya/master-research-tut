import sys
import unittest
import pandas as pd
sys.path.append('../main')
import acceleration_plot3 as ap


class TestAccelerationPlot3(unittest.TestCase):

    def _test_one_column_read_csv_by_pandas(self):
        df_test = ap.read_csv_('./test_dataset/demo.csv')
        df_one_column = pd.DataFrame({'a':[0]})
        pd.testing.assert_frame_equal(df_test, df_one_column)

    def test_real_columns_read_csv_by_pandas(self):
        df_test = ap.read_csv_('./test_dataset/demo.csv')
        df_real_columns = pd.DataFrame({
            'Unnamed: 0':[0],
            'line':[0],
            'time':[0],
            'Acceleration(X)[g]':[0],
            'Acceleration(Y)[g]':[0],
            'Acceleration(Z)[g]':[0],
            'AngularRate(X)[dps]':[0],
            'AngularRate(Y)[dps]':[0],
            'AngularRate(Z)[dps]':[0],
            'Temperature[degree]':[0],
            'Pressure[hPa]':[0],
            'MagnetCount':[0],
            'MagnetSwitch':[0],
            'onCrossroad':[0],
            'crossroadID':[0],
            })
        pd.testing.assert_frame_equal(df_test, df_real_columns)

if __name__ == '__main__':
    unittest.main()
