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

# pragma once

#include "fp.mem.h"
#include "fp.id.h"
#include <opencv2/opencv.hpp>
#include <time.h>
#include "net.h"
#include "mat.h"

namespace hiar_impl {
namespace frontal_profile_faces {

/**
 * @brief Distinguish frontal and profile faces by wrapping Tencent-Ncnn.
 */
class Classifer {
  public:
    /// Minimum width of inputing image.
    const int kTargetWidth = 48;
    /// Minimum height of inputing image.
    const int kTargetHeight = 48;
    /// Normalized parameters for model inputs used by Ncnn.
    const float kNormVals[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};

    /**
     * @brief Construct a new Classifer object.
     */
    Classifer();

    /**
     * @brief Destroy the Classifer object.
     * @remark Release the resource of Ncnn.
     */
    ~Classifer();

    /**
     * @brief Distinguish the face.
     * 
     * @param input[in] Origin image data.
     * @param isRgb[in] Confirm the color channels' order.
     * @return true     Frontal face.
     * @return false    Profile face.
     */
    bool Predict(const cv::Mat &input, bool isRgb);

    /**
     * @brief Get the cost of single predict by millsecond.
     *        @see Classifer::Predict
     * 
     * @param prepare_cost[in, out] Time cost for preparing data. 
     * @param infer_cost[in, out]   Time cost for inferring.
     */
    void GetCostMs(double &prepare_cost, double &infer_cost);

    /**
     * @brief	Loading file line by line.
     * @param file[in] Path to read file.
     * @return         Save lines in std::vector<std::string>.
     */
    static std::vector<std::string> ReadTxt(const std::string file);

    /// Recording the score.
    double score_ = 0;

  private:
    /// Larger than this value is frontal, otherwise is profile.
    const float kThreshold = 0.5;

    /**
     * @brief Loading default model.
     */
    void DefaultInit();

    /**
     * @brief	Converting rectangle into square with fixed center.
     * @param rect[in, out] Inputting rectangle.
     * @note The edge of square is the shorter one
     *       between width and height from rectangle.
     */
    void CenterSquare(cv::Rect &rect);

    /**
     * @brief Convert cv::Mat into ncnn::Mat.
     * @param image[in]   Origin image data.
     * @return ncnn::Mat  Data for inferring.
     * @note There are several pre-processes in the function.
     */
    ncnn::Mat ConvertOneImage(const cv::Mat& image);

    /// Saving the model data.
    ncnn::Net net_;
    /// Saving the inputting data.
    ncnn::Mat one_image_values_;
    /// Saving the outputting data.
    ncnn::Mat out_;
    /// Recording the preparing cost.
    clock_t prepare_cost_ = 0;
    /// Recording the inferring cost.
    clock_t infer_cost_ = 0;
};
}
}