#include "quality_assessment.h"

using namespace hiar_impl::face;

int main() {
  int error = 0;
  cv::Mat image;
  double prepare_cost = 0;
  double infer_cost = 0;
  double prepare_cost_total = 0;
  double infer_cost_total = 0;
  bool rtv = false;

  std::pair<std::string, bool> frontal("/home/devin/MyTmp/Pytmp/pPcList.txt",
                                       true);
  std::pair<std::string, bool> profile("/home/devin/MyTmp/Pytmp/nPcList.txt",
                                       false);
  std::pair<std::string, bool> used = profile;
  QualityAssessment instance = QualityAssessment();
  std::vector<std::string> imageList = QualityAssessment::ReadTxt(used.first);

  for (auto imageName : imageList) {
      if ("" != imageList.at(0)) {
        image = cv::imread(imageName, CV_LOAD_IMAGE_COLOR);
      }

      // std::cout << imageName << std::endl;
      rtv = instance.Predict(image, false);
      if (used.second != rtv) {
        std::cout << imageName << std::endl;
        std::cout << instance.score_ << std::endl;
        error += 1;
      }

      instance.GetCostMs(prepare_cost, infer_cost);
      prepare_cost_total += prepare_cost;
      infer_cost_total += infer_cost;
  }

  std::cout << "Case num: " << imageList.size() << std::endl;
  std::cout << "Error rate: " << float(error) / imageList.size() << std::endl;
  std::cout << "Average time: "
            << (prepare_cost_total + infer_cost_total) / imageList.size()
            << " ms"
            << std::endl;
  std::cout << "Prepare time: "
            << prepare_cost_total / imageList.size()
            << " ms"
            << std::endl;
  std::cout << "Infer time: "
            << infer_cost_total / imageList.size()
            << " ms"
            << std::endl;
  return 0;
}