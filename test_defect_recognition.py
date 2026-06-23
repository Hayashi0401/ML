import os
import unittest
import numpy as np
import cv2

# テスト対象の関数・変数をインポート
# ※defect_recognition.pyが実行時にdatasetフォルダを探す仕様のため、
#  環境に合わせてインポート時のエラーを防ぐ処理を入れるか、同一ディレクトリに配置してください。
import defect_recognition
from defect_recognition import split_images, pred_quality, CLASS_NAMES


class TestDefectRecognition(unittest.TestCase):

    def setUp(self):
        """テスト前の準備"""
        # テスト用のダミー画像（200x200x3、RGBカラー）を作成
        self.dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_split_images_success(self):
        """画像の分割処理（正常系）のテスト"""
        # 10枚のダミー画像リストを作成
        dummy_images = [self.dummy_image] * 10
        test_ratio = 0.2

        train_set, test_set = split_images(dummy_images, test_ratio)

        # 10枚の20%（2枚）がテスト、残りの8枚が学習用になるか検証
        self.assertEqual(len(train_set), 8)
        self.assertEqual(len(test_set), 2)

    def test_split_images_insufficient_data(self):
        """画像が2枚未満の場合にValueErrorを吐くかのテスト"""
        single_image_list = [self.dummy_image]
        
        with self.assertRaises(ValueError):
            split_images(single_image_list, 0.2)

    def test_pred_quality(self):
        """推論関数（pred_quality）が定義されたクラス名のいずれかを返すかテスト"""
        # 実際にモデルがビルド・学習（またはロード）されている前提
        try:
            result = pred_quality(self.dummy_image)
            # 戻り値が ['good', 'oil', 'scratch', 'stain'] のいずれかであることを確認
            self.assertIn(result, CLASS_NAMES)
        except Exception as e:
            self.fail(f"pred_quality の実行中にエラーが発生しました: {e}")


if __name__ == '__main__':
    unittest.main()