import sys
import unittest
import pandas as pd
sys.path.append('../main')
import acceleration_plot3 as ap


class TestAccelerationPlot3(unittest.TestCase):

    def test_read_csv_by_pandas(self):
        df_test = ap.read_csv_('./test_dataset/demo.csv')
        df_one_column = pd.DataFrame({'a':[0]})
        pd.testing.assert_frame_equal(df_test, df_one_column)


if __name__ == '__main__':
    unittest.main()
