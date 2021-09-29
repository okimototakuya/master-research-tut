import os
import glob
import numpy as np
import datetime
import unittest
import pandas as pd
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

    def _setUp(self):
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
        N = 1000            # データ数(個)
        f0 = [x*440 for x in range(N)]
        #f0 = [440*np.random.randint(5) for _ in range(N)]
        amp = [x for x in range(N)]
        #amp = [np.random.randint(5) for _ in range(N)]
        #amp = [1 for _ in range(N)]
        fs = 96000          # 
        addnum = 5.0
        output_type = 'series'  # 関数create_sin_waveの出力型
        def create_sin_wave(sample, input_output_type):
            '''
            sin関数配列を、シーケンスで出力.

            Note
            -----
            amplitude: 振幅
            f0: 
            fs: 
            sample: データ数
            '''
            wave = [(amp[n] * np.sin(2.0 * np.pi * f0[n] * n / fs)) for n in np.arange(sample)]
            if input_output_type == 'list':
                return wave                 # (標準)リスト型
            elif input_output_type == 'dataframe':
                return pd.DataFrame(wave)   # pd.DataFrame型(columns:0, rows:0,1,2,...)
            elif input_output_type == 'series':
                return pd.Series(wave)   # pd.DataFrame型(columns:0, rows:0,1,2,...)
            else:
                raise Exception('関数create_sin_waveに与える引数に誤りがあります.')
        wave = create_sin_wave(N, output_type)
        X = spfft.fft(wave.values.tolist())
        freqList = spfft.fftfreq(N, d=1.0/ fs)
        amplitude = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in X]  # 振幅スペクトル
        # グラフの描画
        fig = plt.figure()
        ax1 = fig.add_subplot(211)  # 波形描画用のサブプロット
        ax2 = fig.add_subplot(212)  # 振幅スペクトル描画用のサブプロット
        ## 波形を描画
        ax1.plot(range(0,N), wave[0:N], label = "wave")
        # 振幅スペクトルを描画
        ax2.plot(freqList, amplitude, marker='.', linestyle='-',label = "fft plot")
        # グラフの出力
        plt.show()

    def _test_pandas_dataframe(self):
        '''
        pandas.DataFrame型の学習用テスト
        '''
        wave = [1.0 * np.sin(2.0 * np.pi * 440 * n  / 96000) for n in np.arange(10)]
        data_frame = pd.DataFrame(wave)
        series = pd.Series(wave)
        #print(data_frame[:5])
        #print([x for x in data_frame.values])
        print(data_frame)
        print(type(data_frame.values))
        print(data_frame.values)
        print(data_frame.values.tolist())
        print(series.values.tolist())


if __name__ == '__main__':
    unittest.main()
