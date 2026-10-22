import os
import unittest
import numpy as np

class TestMapperFeaturesAndModel(unittest.TestCase):
    def test_expand_features_shape(self):
        from modules.mapper import expand_features
        X_raw = np.random.randn(10, 12)
        X_exp = expand_features(X_raw)
        self.assertEqual(X_exp.shape[0], 10)
        self.assertEqual(X_exp.shape[1], 19)

    def test_expand_features_1d(self):
        from modules.mapper import expand_features
        vec_12 = np.random.randn(12)
        vec_19 = expand_features(vec_12)
        self.assertEqual(vec_19.shape[0], 19)

    def test_predict_none_when_uncalibrated(self):
        from modules.mapper import GazeScreenMapper
        mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path="non_existent_model.pkl")
        self.assertIsNone(mapper.predict([0.5] * 12))

    def test_dual_axis_train_and_predict(self):
        from modules.mapper import GazeScreenMapper
        test_model_path = "test_gaze_model.pkl"
        if os.path.exists(test_model_path):
            os.remove(test_model_path)

        mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path=test_model_path)
        np.random.seed(42)
        X = np.random.randn(30, 12)
        Y = np.random.rand(30, 2)
        mapper.X_data = X.tolist()
        mapper.Y_data = Y.tolist()

        mapper.train_model()
        self.assertIsNotNone(mapper.model_x)
        self.assertIsNotNone(mapper.model_y)
        self.assertIsNotNone(mapper.model)

        pred = mapper.predict(X[0].tolist(), timestamp=1.0)
        self.assertIsNotNone(pred)
        self.assertTrue(0 <= pred[0] < 1920)
        self.assertTrue(0 <= pred[1] < 1080)

        if os.path.exists(test_model_path):
            os.remove(test_model_path)

    def test_recenter_offset(self):
        from modules.mapper import GazeScreenMapper
        mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path="test_recenter_model.pkl")
        # Tâm màn hình: 960, 540. Dự đoán thô: 900, 500 -> chênh lệch: +60, +40
        # RECENTER_SMOOTH_FACTOR = 0.5 -> offset = 0.5 * 60 = 30, 0.5 * 40 = 20
        mapper.apply_recenter(pred_raw_x=900, pred_raw_y=500)
        self.assertAlmostEqual(mapper.offset_x, 30.0)
        self.assertAlmostEqual(mapper.offset_y, 20.0)

        # Áp dụng lần 2 để hội tụ gần hơn
        mapper.apply_recenter(pred_raw_x=900, pred_raw_y=500)
        self.assertAlmostEqual(mapper.offset_x, 45.0)

        mapper.reset_offset()
        self.assertEqual(mapper.offset_x, 0.0)
        self.assertEqual(mapper.offset_y, 0.0)

    def test_micro_dead_zone(self):
        from modules.mapper import GazeScreenMapper
        test_model_path = "test_deadzone_model.pkl"
        if os.path.exists(test_model_path):
            os.remove(test_model_path)

        mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path=test_model_path)
        # Giả lập model đã train
        np.random.seed(42)
        X = np.random.randn(20, 12)
        Y = np.zeros((20, 2)) + 0.5  # tất cả nhìn vào giữa màn hình
        mapper.X_data = X.tolist()
        mapper.Y_data = Y.tolist()
        mapper.train_model()

        pt1 = mapper.predict(X[0].tolist(), timestamp=1.0)
        # Thay đổi cực nhỏ feature vector
        micro_jitter = X[0] + 1e-6
        pt2 = mapper.predict(micro_jitter.tolist(), timestamp=1.033)
        # Do thay đổi cực nhỏ (< DEAD_ZONE_RADIUS), tọa độ phải được giữ nguyên
        self.assertEqual(pt1, pt2)

        if os.path.exists(test_model_path):
            os.remove(test_model_path)

if __name__ == '__main__':
    unittest.main()
