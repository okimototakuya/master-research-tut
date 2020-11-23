import os
import glob
import unittest
import matplotlib.pyplot as plt
import pandas as pd


class TestApi(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _test_globassert(self):
        'glob.glob()の返り値をテスト→True/False'
        #assert glob.glob('./test_plot/*.png') is not None
        self.assertTrue(glob.glob('./test_plot/*.png'))

    def test_pltsavefig(self):
        'plt.savefig()がサポートされているかテスト→されている'
        df = pd.DataFrame([[1,2,3],[4,5,6]], columns=['a','b','c'])
        ax = df.plot()
        plt.savefig('./test_plot_api/demo_savefig.png')
        self.assertTrue(glob.glob('./test_plot_api'))


def main():
    if glob.glob('./test_plot/*'):
        print('test_plotディレクトリ下に、ファイルが存在します.')
    else:
        print('ファイルは存在しません.')

    #print(os.path.isfile('./test_plot/*'))
    print(os.path.isfile('test_plot/*'))

if __name__ == '__main__':
    #main()
    unittest.main()
