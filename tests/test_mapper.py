import os
import unittest
import numpy as np

from modules.mapper import expand_features, GazeScreenMapper, EXPANDED_FEATURE_DIM
from modules.config import DEAD_ZONE_RADIUS


class TestMapperFeaturesAndModel(unittest.TestCase):
    def test_expand_features_shape(self):
        X_raw = np.random.randn(10, 12)
        X_exp = expand_features(X_raw)
        self.assertEqual(X_exp.shape[0], 10)
        self.assertEqual(X_exp.shape[1], EXPANDED_FEATURE_DIM)
        self.assertEqual(EXPANDED_FEATURE_DIM, 12)

    def test_expand_features_1d(self):
        vec_12 = np.random.randn(12)
        vec_exp = expand_features(vec_12)
        self.assertEqual(vec_exp.shape[0], 12)

    def test_predict_none_when_uncalibrated(self):
        mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path="non_existent_model.pkl")
        self.assertIsNone(mapper.predict([0.5] * 12))

    def test_dual_axis_train_and_predict(self):
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
        mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path="test_recenter_model.pkl")
        # Giả lập dự đoán thô là (900, 500), tâm màn hình là (960, 540)
        # Chênh lệch: +60, +40
        mapper.apply_recenter(pred_raw_x=900, pred_raw_y=500)
        self.assertEqual(mapper.offset_x, 60)
        self.assertEqual(mapper.offset_y, 40)

        mapper.reset_offset()
        self.assertEqual(mapper.offset_x, 0.0)
        self.assertEqual(mapper.offset_y, 0.0)

    def test_recenter_with_feature_vector(self):
        test_model_path = "test_recenter_fv_model.pkl"
        if os.path.exists(test_model_path):
            os.remove(test_model_path)

        mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path=test_model_path)
        np.random.seed(42)
        X = np.random.randn(20, 12)
        Y = np.zeros((20, 2)) + 0.4
        mapper.X_data = X.tolist()
        mapper.Y_data = Y.tolist()
        mapper.train_model()

        mapper.recenter(feature_vector=X[0].tolist())
        self.assertNotEqual(mapper.offset_x, 0.0)
        self.assertNotEqual(mapper.offset_y, 0.0)

        if os.path.exists(test_model_path):
            os.remove(test_model_path)

    def test_dead_zone_jitter_stabilization(self):
        """Kiểm tra Dead-Zone loại bỏ hoàn toàn rung giật vi mô khi mắt nhìn cố định."""
        test_model_path = "test_deadzone_model.pkl"
        if os.path.exists(test_model_path):
            os.remove(test_model_path)

        mapper = GazeScreenMapper(screen_w=1920, screen_h=1080, model_path=test_model_path)
        np.random.seed(42)
        X = np.random.randn(20, 12)
        Y = np.zeros((20, 2)) + 0.5
        mapper.X_data = X.tolist()
        mapper.Y_data = Y.tolist()
        mapper.train_model()

        pt1 = mapper.predict(X[0].tolist(), timestamp=1.0)
        self.assertIsNotNone(pt1)

        # Rung vi mô rất nhỏ (< DEAD_ZONE_RADIUS sau khi qua bộ lọc)
        jitter_fv = (np.array(X[0]) + 1e-5).tolist()
        pt2 = mapper.predict(jitter_fv, timestamp=1.033)

        self.assertEqual(pt1, pt2)

        if os.path.exists(test_model_path):
            os.remove(test_model_path)


if __name__ == '__main__':
    unittest.main()
