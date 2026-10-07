#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

inline std::vector<std::vector<double>> read_csv_file(const std::string& filename) {
  std::vector<std::vector<double>> data;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return data;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::vector<double> row;
    std::stringstream ss(line);
    std::string value;
    while (std::getline(ss, value, ',')) {
      if (!value.empty()) row.push_back(std::stod(value));
    }
    if (!row.empty()) data.push_back(std::move(row));
  }
  return data;
}
