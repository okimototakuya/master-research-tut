import os
import glob
import unittest


class TestApi(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_globassert(self):
        #assert glob.glob('./test_plot/*.png') is not None
        self.assertTrue(glob.glob('./test_plot/*.png'))

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
