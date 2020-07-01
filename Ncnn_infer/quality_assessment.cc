/**
 * @file
 * @author	Devin dai
 * @version	0.0.1
 * @date	2020/06/11
 * @brief	Deep-learning for distinguishing frontal and profile faces.
 * @details	Use square roi to distinguish frontal and profile faces
 *          based on Tencent-Ncnn.
 * @section LICENSE
 * Copyright 2020 Hiscene
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * 		http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "quality_assessment.h"

namespace hiar_impl {
namespace face {

QualityAssessment::QualityAssessment() {
  DefaultInit();
  return;
}

QualityAssessment::~QualityAssessment() {
  net_.clear();
  return;
}

bool QualityAssessment::Predict(const cv::Mat &image, bool isRgb) {
  cv::Mat rgb;
  clock_t start = 0;
  clock_t end = 0;

  if (image.empty()
      || 3 != image.channels()
      || kTargetWidth > image.cols
      || kTargetHeight > image.rows
      || isRgb) {
    std::cout << "Invalid image !!!" << std::endl;
    assert(false);
    return false;
  }
  
  if (kIsRecording) {
    start = clock();
    one_image_values_ = ConvertOneImage(image);
    end = clock();
    prepare_cost_ = end - start;

    start = clock();
    ncnn::Extractor infer = net_.create_extractor();
    infer.set_light_mode(true);
    infer.input(ncnn_param_id::LAYER_input_1_0, one_image_values_);
    infer.extract(ncnn_param_id::BLOB_Identity_0, out_);
    end = clock();
    infer_cost_ = end - start;
  } else {
      one_image_values_ = ConvertOneImage(image);
      ncnn::Extractor infer = net_.create_extractor();
      infer.set_light_mode(true);
      infer.input(ncnn_param_id::LAYER_input_1_0, one_image_values_);
      infer.extract(ncnn_param_id::BLOB_Identity_0, out_);
  }

  // std::cout << "out_.w: " << out_.w << std::endl;
  // std::cout << out_[0] << std::endl;
  score_ = out_[0];
  return kThreshold < out_[0];
}

void QualityAssessment::GetCostMs(double &prepare_cost, double &infer_cost) {
  prepare_cost = 1000.f * prepare_cost_ / (CLOCKS_PER_SEC);
  infer_cost = 1000.f * infer_cost_ / (CLOCKS_PER_SEC);
  prepare_cost_ = 0;
  infer_cost_ = 0;
  return;
}

float QualityAssessment::GetBlurScore(const cv::Mat &origin_image,
                                      const cv::Rect &face_rect,
                                      float* landmarks, int landmarks_len) {
	float rtv = 0;
  cv::Mat blurGray;
  cv::Mat blurAnalysis;
  cv::Mat mean;
  cv::Mat stdDev;
  cv::Rect landMarkRect = GetRoi(origin_image, face_rect, landmarks, 
                                 landmarks_len);
	if (0 >= landMarkRect.area()) {
		return 0;
	}

	cv::Mat faceRoi = origin_image(landMarkRect);
  cv::cvtColor(faceRoi, blurGray, cv::COLOR_BGR2GRAY);
  
  if ((faceRoi.cols * faceRoi.rows) > kExpectedPixels) {
      float ratio = sqrt(kExpectedPixels / (faceRoi.cols * faceRoi.rows));
      cv::resize(blurGray, blurGray, cv::Size(0, 0), ratio, ratio,
                 cv::INTER_NEAREST);
  }

  // std::cout << "blurGray.width: " << blurGray.cols << std::endl;
  // std::cout << "blurGray.height: " << blurGray.rows << std::endl;
  cv::Laplacian(blurGray, blurAnalysis, CV_64F);
  cv::meanStdDev(blurAnalysis, mean, stdDev);
  rtv = stdDev.at<double>(0);
  // std::cout << "BlurScore: " << rtv << std::endl;

  // cv::normalize(blurAnalysis, blurAnalysis, 0, 1, cv::NORM_MINMAX);
  // cv::imshow("blurAnalysis", blurAnalysis);
  // cv::imshow("crop4", faceRoi);
  // cv::waitKey();
  return rtv;
}

std::vector<std::string> QualityAssessment::ReadTxt(const std::string file) {
  std::vector<std::string> rtv;
  std::ifstream infile; 
  std::string s;

  infile.open(file.data());
  if (infile.is_open()) {
    while (getline(infile, s)) {
        rtv.push_back(s);
        // std::cout << s << std::endl;
    }
    infile.close();
  } else {
    std::cout << "Loading file failed !!!" << std::endl;
    std::cout << file << std::endl;
    assert(false);
  }
  return rtv;
}

void QualityAssessment::DefaultInit() {
  net_.load_param(ncnn_param_bin);
  net_.load_model(ncnn_bin);
  // net_.load_param("ncnn.param");
  // net_.load_model("ncnn.bin");
  return;
}

void QualityAssessment::CenterSquare(cv::Rect &rect) {
  if (0 > rect.width || 0 > rect.height) {
    std::cout << "Invaild cv::Rect !!!" << std::endl;
    assert(false);
    return;
  }

  int diff = rect.width - rect.height;
  if (0 > diff) {
      diff = -diff;
      rect.y += int(diff / 2);
      rect.height -= diff;
  } else {
      rect.x += int(diff / 2);
      rect.width -= diff;  
  }
  return;
}

ncnn::Mat QualityAssessment::ConvertOneImage(const cv::Mat& image) {
  cv::Rect roi(0, 0, image.cols, image.rows);
  CenterSquare(roi);
  cv::Mat newImg = image(roi);
  cv::resize(newImg, newImg, cv::Size(kTargetWidth, kTargetHeight));
  ncnn::Mat rtv = ncnn::Mat::from_pixels(newImg.data, ncnn::Mat::PIXEL_BGR2RGB,
                                         newImg.cols, newImg.rows);
  rtv.substract_mean_normalize(0, kNormVals);
  return rtv;
}

cv::Rect QualityAssessment::GetRoi(const cv::Mat &origin_image,
                                   const cv::Rect &face_rect,
				                           float* landmarks, int landmarks_len) {
	cv::Rect landMarkRect(face_rect.width, face_rect.height, 0, 0);
	for (int i = 0; i < landmarks_len; i++) {
		int xW = face_rect.x + int(landmarks[i] * face_rect.width);
		int yH = face_rect.y + int(landmarks[i + landmarks_len] * face_rect.height);
		landMarkRect.x = landMarkRect.x < xW ? landMarkRect.x : xW;
		landMarkRect.y = landMarkRect.y < yH ? landMarkRect.y : yH;
		landMarkRect.width = landMarkRect.width > xW ? landMarkRect.width : xW;
		landMarkRect.height = landMarkRect.height > yH ? landMarkRect.height : yH;
	}
	landMarkRect.width -= landMarkRect.x;
	landMarkRect.height -= landMarkRect.y;

  RectCenterScale(landMarkRect, kExpandingRatio);
	landMarkRect = landMarkRect
                 & cv::Rect(0, 0, origin_image.cols, origin_image.rows);
	return landMarkRect;
}

void QualityAssessment::RectCenterScale(cv::Rect &rect, const float ratio) {
	if (0 > ratio) {
    std::cout << "Expanding ratio error !!!" << std::endl;
    assert(false);
	} else {
		int deltaW = cvRound((ratio - 1) * rect.width);
		int deltaH = cvRound((ratio - 1) * rect.height);
		cv::Point pt;
		pt.x = cvRound(deltaW / 2.0);
		pt.y = cvRound(deltaW / 2.0);
		rect.width += deltaW;
		rect.height += deltaH;
	}
  return;
}

}
}