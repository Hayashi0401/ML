import os
import io
import json
import unittest

# =====================================================================
# 🤫 TensorFlowやシステム全体の不要な警告ログ（WARNING）を完全に非表示
# =====================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from main import app, UPLOAD_FOLDER

TARGET_FILENAME = "test.png"


class TestFlaskWebApp(unittest.TestCase):
    # 📋 各テストの合格(True) / 不合格(False) とエラー文を確実に保存するリスト
    test_results = []
    ui_errors = []

    def setUp(self):
        """各テストの実行前に呼ばれるセットアップルーティン"""
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SECRET_KEY'] = 'test-secret-key'
        self.client = app.test_client()
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    def test_01_index_get(self):
        """トップページへのGETリクエストが成功するかテスト"""
        title = "トップページ（画面）のアクセス確認テスト"
        print(f"【{title}】確認中...")
        response = self.client.get('/')
        if response.status_code != 200:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
            return
        print(f"【{title}】→ OK\n")
        TestFlaskWebApp.test_results.append(True)

    def test_02_index_post_no_file(self):
        """ファイルなしでPOSTした場合のリダイレクト検証"""
        title = "ファイル未選択時のエラー・リダイレクトテスト"
        print(f"【{title}】確認中...")
        response = self.client.post('/', data={})
        if response.status_code != 302:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
            return
        print(f"【{title}】→ OK\n")
        TestFlaskWebApp.test_results.append(True)

    def test_03_file_upload_and_prediction(self):
        """ダミー画像を実際にアップロードして200 OKが返るかテスト"""
        title = "画像判定システム（AIモデル）の動作テスト"
        print(f"【{title}】確認中...")
        img_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        data = {'file': (io.BytesIO(img_bytes), 'test_defect.png')}
        response = self.client.post('/', data=data, content_type='multipart/form-data')
        if response.status_code != 200:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
            return
        print(f"【{title}】→ OK\n")
        TestFlaskWebApp.test_results.append(True)

    def test_04_sample_image_upload_and_ui_elements(self):
        """UI・分類結果の画面表示テスト"""
        title = f"サンプル画像({TARGET_FILENAME})によるUI・画面表示テスト"
        print(f"【{title}】確認中...")
        img_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        image_file = io.BytesIO(img_bytes)
        data = {'file': (image_file, TARGET_FILENAME)}

        response = self.client.post('/', data=data, content_type='multipart/form-data')
        if response.status_code != 200:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            TestFlaskWebApp.ui_errors.append(f"❌ 判定ボタンの押下（POSTリクエスト）に失敗しました。Status: {response.status_code}")
            self.fail()
            return

        html_content = response.data.decode('utf-8')
        has_error = False

        # 【確認①】画像とimgタグのチェック（失敗時のみ出力）
        if (TARGET_FILENAME not in html_content) or ('<img' not in html_content):
            TestFlaskWebApp.ui_errors.append(f"❌ 【確認①】画面のHTML内に、ファイル名「{TARGET_FILENAME}」または <img> タグが見つかりません。")
            has_error = True

        # 【確認②】分類結果領域のチェック（成功時は表示しない）
        if not any(keyword in html_content for keyword in ['結果', '判定', 'result', 'prediction', '分類']):
            TestFlaskWebApp.ui_errors.append("❌ 【確認②】画面のHTML内に、判定結果を示すキーワード（結果、判定、分類など）が見つかりません。")
            has_error = True

        # 【確認③】信頼度（確率）のチェック（失敗時のみ出力）
        if not any(keyword in html_content for keyword in ['信頼度', '確率', '％', '%', 'confidence']):
            TestFlaskWebApp.ui_errors.append("❌ 【確認③】画面のHTML内に、判定の「信頼度（確率）」の数値・文言が見つかりません。")
            has_error = True

        if has_error:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
        else:
            print(f"【{title}】→ OK\n")
            TestFlaskWebApp.test_results.append(True)

    def test_05_gradcam_success_heatmap(self):
        """ヒートマップ（Grad-CAM）が正常に応答を返すかテスト"""
        title = "ヒートマップ（Grad-CAM）画像出力テスト"
        print(f"【{title}】確認中...")
        payload = {'filename': TARGET_FILENAME}
        response = self.client.post('/gradcam', data=json.dumps(payload), content_type='application/json')
        if response.status_code != 200:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
            return
        
        if 'application/json' in response.content_type:
            data = json.loads(response.data.decode('utf-8'))
            if 'gradcam_image' not in data:
                print(f"【{title}】→ ❌\n")
                TestFlaskWebApp.test_results.append(False)
                self.fail()
                return
        else:
            if 'image' not in response.content_type:
                print(f"【{title}】→ ❌\n")
                TestFlaskWebApp.test_results.append(False)
                self.fail()
                return
        print(f"【{title}】→ OK\n")
        TestFlaskWebApp.test_results.append(True)

    def test_06_get_history_valid_class(self):
        """有効なクラス名での履歴取得APIのテスト"""
        title = "過去の判定履歴データ取得テスト"
        print(f"【{title}】確認中...")
        response = self.client.get('/history/good')
        if response.status_code != 200:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
            return
        
        data = json.loads(response.data.decode('utf-8'))
        if 'files' not in data:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
            return
        print(f"【{title}】→ OK\n")
        TestFlaskWebApp.test_results.append(True)

    def test_07_get_history_invalid_class(self):
        """無効なクラス名での履歴取得APIのエラーハンドリングテスト"""
        title = "不正なカテゴリ名指定時の履歴取得エラーテスト"
        print(f"【{title}】確認中...")
        response = self.client.get('/history/invalid_class_name')
        if response.status_code != 400:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
            return
        print(f"【{title}】→ OK\n")
        TestFlaskWebApp.test_results.append(True)

    def test_08_gradcam_no_image_error(self):
        """画像がまだない状態でGrad-CAMを要求した際のエラーテスト"""
        title = "画像データ未存在時のGrad-CAMエラーハンドリングテスト"
        print(f"【{title}】確認中...")
        response = self.client.post('/gradcam', data=json.dumps({}), content_type='application/json')
        if response.status_code != 200:
            print(f"【{title}】→ ❌\n")
            TestFlaskWebApp.test_results.append(False)
            self.fail()
            return
        print(f"【{title}】→ OK\n")
        TestFlaskWebApp.test_results.append(True)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlaskWebApp)
    
    import sys
    f = open(os.devnull, 'w')
    runner = unittest.TextTestRunner(stream=f)
    runner.run(suite)
    f.close()

    # 🛑 【集計結果表示】
    total_tests = 8
    passed_tests = TestFlaskWebApp.test_results.count(True)
    unique_errors = sorted(list(set(TestFlaskWebApp.ui_errors)))

    print("----------------------------------------------------------------------")
    print(f"{passed_tests}/{total_tests} OK")
    
    if unique_errors:
        error_count = len(unique_errors)
        print(f"{error_count}個のエラーが確認されました")
        print("----------------------------------------------------------------------")
        for err in unique_errors:
            print(err)
    print("----------------------------------------------------------------------")