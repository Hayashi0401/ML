import os
import io
import json
import unittest
from main import app, UPLOAD_FOLDER  # main.pyからFlaskインスタンス等をインポート


class TestFlaskWebApp(unittest.TestCase):

    def setUp(self):
        """各テストの実行前に呼ばれるセットアップルーティン"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

        # テスト用アップロードフォルダの設定（必要に応じて）
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    def tearDown(self):
        """各テストの実行後にクリーンアップするルーティン"""
        # 必要に応じてテスト中に生成されたファイルを削除する処理をここに記述します
        pass

    def test_index_get(self):
        """トップページへのGETリクエストが成功するかテスト"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # HTML内に特定の文字列が含まれているかチェック（index.htmlの内容に合わせる）
        # self.assertIn(b'html', response.data)

    def test_index_post_no_file(self):
        """ファイルなしでPOSTした場合のリダイレクト検証"""
        response = self.client.post('/', data={})
        # ファイルがない場合は通常リダイレクト（302）されるロジックの検証
        self.assertEqual(response.status_code, 302)

    def test_file_upload_and_prediction(self):
        """ダミー画像を実際にアップロードして200 OKが返るかテスト"""
        # テスト用のダミー画像（1x1ピクセルのPNG）をバイナリデータとして作成
        img_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        
        data = {
            'file': (io.BytesIO(img_bytes), 'test_defect.png')
        }
        
        response = self.client.post('/', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)

    def test_get_history_valid_class(self):
        """有効なクラス名での履歴取得APIのテスト"""
        response = self.client.get('/history/good')
        self.assertEqual(response.status_code, 200)
        
        # レスポンスがJSONで、'files' キーを持っているか検証
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('files', data)

    def test_get_history_invalid_class(self):
        """無効なクラス名での履歴取得APIのエラーハンドリングテスト"""
        response = self.client.get('/history/invalid_class_name')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn('error', data)

    def test_gradcam_no_image_error(self):
        """画像がまだない状態でGrad-CAMを要求した際のエラーテスト"""
        # 空のJSONでリクエストを投げる
        response = self.client.post(
            '/gradcam',
            data=json.dumps({}),
            content_type='application/json'
        )
        # 画像がないため 400 Bad Request になることを想定
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()