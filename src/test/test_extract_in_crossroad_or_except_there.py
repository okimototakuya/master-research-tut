import unittest
import filecmp
import os
import glob
import sys
sys.path.append('../build')
import extract_in_crossroad_or_except_there as ece
import pickup_cross_label as pcl


class TestExtractCrossroad(unittest.TestCase):
    '''
    Pythonスクリプトextract_in_crossroad_or_except_there.pyについてテスト
    '''
    def setUp(self):
        '''
        Pythonスクリプトpickup_cross_label.pyから、交差点ラベルのリストを取得
        '''
        self.cross_list = pcl.main()

    def tearDown(self):
        '''
        作成したテストcsvファイルを削除
        '''
        file_list = glob.glob('test-build/hoge_hoge*.csv')
        for file in file_list:
            print('remove: {0}'.format(file))
            os.remove(file)

    def test_extract_in_no_crossroad_right(self):
        '''
        正しく交差点不在時のデータ点を抽出できたかテスト
        '''
        ece.bool_is_oncrossroad = False     # 交差点不在時を抽出
        ece.main()
        for i in range(1, len(self.cross_list)+2):
            self.assertTrue(filecmp.cmp('test-build/hoge_hoge{i}.csv'.format(i=i), 'test-build/in_no_crossroad{i}.csv'.format(i=i)))

    def test_extract_in_crossroad_right(self):
        '''
        正しく交差点滞在時のデータ点を抽出できたかテスト
        '''
        ece.bool_is_oncrossroad = True     # 交差点不在時を抽出
        ece.main()
        for i in range(1, len(self.cross_list)+2):
            self.assertTrue(filecmp.cmp('test-build/hoge_hoge{0}.csv'.format(i), '../../dataset/{0}crossroad.csv'.format(self.cross_list[i])))


if __name__ == '__main__':
    unittest.main()
