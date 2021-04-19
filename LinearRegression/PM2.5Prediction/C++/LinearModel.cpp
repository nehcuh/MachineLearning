//
// Created by 胡琛 on 2021/4/9.
//

#include "LinearModel.h"

/**
 * 读取 csv 文件，导入特征张量与标签向量
 *
 * @param file_name [in] 输入的 csv 文件名
 *
 */
void LinearModel::read_csv(const char *file_name) {

  std::fstream fin(file_name, std::ios::in);

  if (fin.fail()) {
    std::cout << "file " << file_name << " not found." << '\n';
  } else {
    std::string tmp;
    std::stringstream ss;

    std::string line;
    std::getline(fin, line); // skip firt row which is header

    while (std::getline(fin, line)) {
      std::vector<double> line_vector;
      ss.clear();
      ss << line;
      line_vector.push_back(1.0);
      while (std::getline(ss, tmp, ',')) {
        line_vector.emplace_back(strtod(tmp.c_str(), nullptr));
      }

      _labels.push_back(line_vector.back());
      line_vector.pop_back();
      _features.emplace_back(std::move(line_vector));
    }
  }
}

/**
 * 对特征张量进行转置操作
 *
 */
void LinearModel::transpose_features() {
  for (std::vector<double>::size_type idx = 0; idx < _features.front().size();
       ++idx) {
    std::vector<double> feature_column;
    for (auto itr = _features.begin(); itr != _features.end(); ++itr) {
      feature_column.push_back((*itr)[idx]);
    }
    _features_transpose.emplace_back(std::move(feature_column));
  }
}

/**
 * 对特征张量进行标准化，这里使用 X-mu / sigma 的方式按列进行标准化
 *
 *
 */
void LinearModel::standard_features() {
  // 对每个特征标准化对应对特征张量每列进行标准化
  // 对应对特征张量转置每行进行标准化
  std::vector<std::tuple<double, double>> mu_sigma{};
  mu_sigma.reserve(_features_transpose.size());
  // 对 bias 标准化直接令其为 0.
  std::for_each(std::begin(_features_transpose.front()),
                std::end(_features_transpose.front()),
                [&](double &d) { d = 0.; });
  for (auto itr = _features_transpose.begin() + 1;
       itr != _features_transpose.end(); ++itr) {
    double sum = std::accumulate(std::begin(*itr), std::end(*itr), 0.);
    double mu = sum / (*itr).size();
    double sigma = 0.;
    std::for_each(std::begin(*itr), std::end(*itr),
                  [&](const double &d) { sigma += ((d - mu) * (d - mu)); });
    sigma = std::sqrt(sigma);
    mu_sigma.push_back({mu, sigma});
    // 对转置特征张量进行标准化
    std::for_each(std::begin(*itr), std::end(*itr),
                  [&](double &d) { d = (d - mu) / sigma; });
  }

  for (auto itr = _features.begin(); itr != _features.end(); ++itr) {
    (*itr)[0] = 0.;
    for (auto idx = 1; idx < (*itr).size(); ++idx) {
      (*itr)[idx] = ((*itr)[idx] - std::get<0>(mu_sigma[idx])) /
                    std::get<1>(mu_sigma[idx]);
    }
  }
}

/**
 * 模型参数初始化
 *
 */
void LinearModel::init_weights_bias() {
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::normal_distribution<double> distribution(0, 0.1);
  // bias 收缩进广义 weights 向量
  for (auto idx = 0; idx < (*_features.begin()).size(); ++idx) {
    _weights.push_back(distribution(engine));
  }
}

/**
 * 线性模型 Y = XW + bias
 *
 */
void LinearModel::predict_Y() {
  if (_pred_labels.size() > 0) {
    _pred_labels.clear();
  }
  // 检查 features 与 weights 形状, 注意，这里的 weights 为广义 weights，相比
  // features 数要多 1 维
  if ((_features.front()).size() != _weights.size()) {
    std::cout << "当前输入特征与权重向量形状不一致，特征形状为: "
              << _features.size() << "x" << _features.front().size()
              << ", 权重形状为: " << _weights.size() << '\n';
  }

  for (auto itr = _features.begin(); itr != _features.end(); ++itr) {
    double element = 0.;
    for (std::vector<double>::size_type idx = 0; idx < _features.front().size();
         ++idx) {
      element += (*itr)[idx] * _weights[idx];
    }
    _pred_labels.emplace_back(std::move(element));
  }
}

/**
 * 损失函数计算, (pred_Y - Y)^T.dot(pred_Y - Y)
 *
 * @param pred_labels 预测标签值
 * @param true_labels 实际标签值
 */
void LinearModel::cal_loss() {
  if (_losses.size() > 0) {
    _losses.clear();
  }
  // 检查预测值与真值形状
  if (_pred_labels.size() != _labels.size()) {
    std::cout << "预测标签值与实际标签值形状不一致，预测标签值形状为: "
              << _pred_labels.size() << ", 实际标签值形状为: " << _labels.size()
              << '\n';
  }
  for (std::vector<double>::size_type idx = 0; idx < _pred_labels.size();
       ++idx) {
    _losses.emplace_back(_pred_labels[idx] - _labels[idx]);
  }
}

/**
 *
 * 计算梯度
 *
 */
void LinearModel::cal_gradient() {
  if (_gradients.size() > 0) {
    _gradients.clear();
  }
  // 判断 X^T 与 loss 形状
  if ((_features_transpose.front()).size() != _losses.size()) {
    std::cout << "输入的特征张量形状与损失形状不一致\n"
              << "特征张量转置形状为: " << _features_transpose.size()
              << "损失的形状为: " << _losses.size() << '\n';
  }

  // 这里计算广义特征张量的 gradients
  for (auto itr = _features_transpose.begin(); itr != _features_transpose.end();
       ++itr) {
    double tmp = 0.;
    for (auto idx = 0; idx < (*itr).size(); ++idx) {
      tmp += (*itr)[idx] * _losses[idx];
    }
    _gradients.emplace_back(std::move(tmp));
  }
}

/**
 * 梯度优化
 *
 * @param lr 学习率
 *
 */
void LinearModel::update_weights(double lr = 0.002) {

  // 形状判断
  if (_gradients.size() != _weights.size()) {
    std::cout << "错误，梯度形状与权重形状不一致\n"
              << "梯度形状为: " << _gradients.size() << '\t'
              << "权重形状为: " << _weights.size() << '\n';
  }

  for (auto idx = 0; idx < _weights.size(); ++idx) {
    _weights[idx] -= lr * _gradients[idx];
  }
}

/**
 * 训练过程
 *
 *
 */
void LinearModel::train(int epochs = 2000) {
  // 导入特征张量与标签向量
  read_csv("/Users/huchen/Documents/Projects/MachineLearning/LinearRegression/"
           "Data/train_set.csv");

  // 计算特征张量转置
  transpose_features();

  // 特征张量标准化
  standard_features();

  // 初始化模型参数
  init_weights_bias();

  // 初始预测
  predict_Y();
  // 计算初始损失
  cal_loss();

  for (auto epoch = 0; epoch < epochs; ++epoch) {
    if (epoch % 100 == 0) {
      double square_loss = 0.;
      for (auto idx = 0; idx < _losses.size(); ++idx) {
        square_loss += (_losses[idx]) * (_losses[idx]);
      }
      std::cout << "当前训练错误为: " << square_loss / _losses.size() / 2.0
                << '\n';
    }
    cal_gradient();
    update_weights();
    predict_Y();
    cal_loss();
  }
  std::cout << "bias: " << _weights[0] << std::endl;
  predict_Y();
  std::cout << "current prediction is: " << _pred_labels[0]
            << "\n true label is: " << _labels[0] << '\n';
}
