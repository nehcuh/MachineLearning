//
// Created by 胡琛 on 2021/4/9.
//

#ifndef LINEARREGRESSION_LINEARMODEL_H
#define LINEARREGRESSION_LINEARMODEL_H

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

class LinearModel {
private:
  // 特征张量
  std::vector<std::vector<double>> _features;
  std::vector<std::vector<double>> _features_transpose;
  // 标签向量
  std::vector<double> _labels;
  // 预测标签向量
  std::vector<double> _pred_labels;
  // 广义权重 (包括 bias)
  std::vector<double> _weights;
  // 损失
  std::vector<double> _losses;
  // 梯度
  std::vector<double> _gradients;

public:
  // 文件读取
  void read_csv(const char *file_name);

  // 张量转置
  void transpose_features();

  // 特征张量标准化
  void standard_features();

  // 参数初始化
  void init_weights_bias();

  // 模型定义
  void predict_Y();

  // 损失函数
  void cal_loss();

  // 梯度计算
  void cal_gradient();

  // 优化方法
  void update_weights(double);

  // 训练
  void train(int);

public:
  inline LinearModel() = default;
};

#endif // LINEARREGRESSION_LINEARMODEL_H
