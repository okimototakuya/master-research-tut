import os
import glob
import numpy as np
import datetime
import unittest
import scipy.fftpack as spfft
import matplotlib.pyplot as plt


class IterAdd():
    '''
    テスト用のイテレータプロトコル
    '''
    def __init__(self, input_date_time):
        self.date_time = input_date_time

    def __iter__(self):
        for _ in range(5):
            self.date_time = self.date_time + datetime.timedelta(microseconds=np.random.binomial(1, 0.6)*100000)
            yield self.date_time.strftime('%M:%S.%f')


class TestApi(unittest.TestCase):

    def setUp(self):
        '''
        イテレータプロトコルの簡易出力テスト
        '''
        date_time = datetime.datetime(2018, 12, 19, 14, minute=00, second=00, microsecond=0)
        iter_add = IterAdd(date_time)
        for i in iter_add:
            print(i)

    def tearDown(self):
        pass

    def _test_os_path_isfile(self):
        '''
        os.path.isfileの学習用テスト
        '''
        self.assertTrue(os.path.isfile('test_plot/*'))

    def _test_glob_glob(self):
        '''
        glob.globの学習用テスト

        Notes
        -----
        書き方その１: unittestでなく、標準のassert文を利用
        assert glob.glob('./test_plot/*.png') is not None  # 的を得ていない書き方。Noneはシングルトンオブジェクト。
        assert glob.glob('./test_plot/*.png')
        書き方その２: unittestのassertメソッド(assertTrue)を利用
        '''
        self.assertTrue(glob.glob('./test_plot/*.png'))

    def test_scipy_fftpack(self):
        '''
        scipy.fftpackの学習用テスト
        '''
        f0 = 440
        fs = 96000
        N = 1000
        addnum = 5.0
        def create_sin_wave(amplitude, f0, fs, sample):
            '''
            sin関数配列を、シーケンスで出力.
            '''
            wave_table = [] # (標準)リスト型
            for n in np.arange(sample):
                sine = amplitude * np.sin(2.0 * np.pi * f0 * n / fs)
                wave_table.append(sine)
            #return wave_table   # (標準)リスト型
            wave_table1 = wave_table
            wave_table2 = [amplitude * np.sin(2.0 * np.pi * f0 * n / fs) for n in np.arange(sample)]
            self.assertEqual(wave_table1, wave_table2)
            #return [amplitude * np.sin(2.0 * np.pi * f0 * n / fs) for n in np.arange(sample)]
            #return pd.DataFrame(    # pd.DataFrame型
            #            [amplitude * np.sin(2.0 * np.pi * f0 * n / fs) for n in np.arange(sample)]
            #        )
        #wave1 = create_sin_wave(1.0, f0, fs, N)
        #X = spfft.fft(wave1[0:N])
        #freqList = spfft.fftfreq(N, d=1.0/ fs)
        #amplitude = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]  # 振幅スペクトル
        ## グラフの描画
        #fig = plt.figure()
        #ax1 = fig.add_subplot(211)  # 波形描画用のサブプロット
        #ax2 = fig.add_subplot(212)  # 振幅スペクトル描画用のサブプロット
        ### 波形を描画
        #ax1.plot(range(0,N), wave1[0:N],label = "wave1")
        ## 振幅スペクトルを描画
        #ax2.plot(freqList, amplitude, marker='.', linestyle='-',label = "fft plot")
        ## グラフの出力
        #plt.show()


if __name__ == '__main__':
    unittest.main()
