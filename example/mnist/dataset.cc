#include "example/mnist/dataset.h"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using SN3PascalVincentType = MNISTDataset::SN3PascalVincentType;
using SN3PascalVincentFile = MNISTDataset::SN3PascalVincentFile;

// Map SN3 type id to enum.
const std::unordered_map<int, SN3PascalVincentType> kTypeMap = {
    {8, SN3PascalVincentType::kUINT8},  {9, SN3PascalVincentType::kINT8},     {11, SN3PascalVincentType::kINT16},
    {12, SN3PascalVincentType::kINT32}, {13, SN3PascalVincentType::kFLOAT32}, {14, SN3PascalVincentType::kFLOAT64},
};

// Byte size of each SN3 data type.
const std::unordered_map<SN3PascalVincentType, size_t> kSN3TypeToSize = {
    {SN3PascalVincentType::kUINT8, 1}, {SN3PascalVincentType::kINT8, 1},    {SN3PascalVincentType::kINT16, 2},
    {SN3PascalVincentType::kINT32, 4}, {SN3PascalVincentType::kFLOAT32, 4}, {SN3PascalVincentType::kFLOAT64, 8},
};

// Map SN3 type to Tensor::DataType.
const std::unordered_map<SN3PascalVincentType, DataType> kSN3TypeToDataType = {
    {SN3PascalVincentType::kUINT8, DataType::kUINT8},     {SN3PascalVincentType::kINT8, DataType::kINT8},
    {SN3PascalVincentType::kINT16, DataType::kINT16},     {SN3PascalVincentType::kINT32, DataType::kINT32},
    {SN3PascalVincentType::kFLOAT32, DataType::kFLOAT32}, {SN3PascalVincentType::kFLOAT64, DataType::kFLOAT64},
};

// MNIST file prefixes.
constexpr char kTrainPrefix[] = "train";
constexpr char kTestPrefix[] = "t10k";

// Read N bytes from a binary stream.
std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

// Bytes per sample.
size_t BytesPerSample(const SN3PascalVincentFile &file) {
    CHECK(!file.dims.empty());
    CHECK_GT(file.dims[0], 0);
    return file.tensor.SizeInBytes() / static_cast<size_t>(file.dims[0]);
}
SN3PascalVincentFile ReadSN3PascalVincentFile(const std::string &path) {
    /*
      magic                                     | dims               | data    |
      reserved | reserved | type_int | num_dims |
      1 byte   | 1 byte   | 1 byte   | 1 byte   | 4*{num_dims} bytes | # bytes |
    */

    // Fail fast if the file does not exist.
    if (!std::filesystem::exists(path)) {
        LOG(FATAL) << "File not found: " << path;
    }

    SN3PascalVincentFile sn3_file;
    std::ifstream ifs(path, std::ios::binary);
    // magic[2] = type, magic[3] = number of dims.
    const auto magic = ReadSeveralBytesFromIfstream(4, &ifs);
    const int num_dims = magic[3];
    const int type_int = magic[2];
    sn3_file.type = kTypeMap.at(type_int);

    // Read shape dims in big-endian order.
    auto &dims = sn3_file.dims;
    dims.resize(num_dims, 0);
    for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
        for (const auto &v : ReadSeveralBytesFromIfstream(4, &ifs)) {
            dims[dim_idx] <<= 8;
            dims[dim_idx] += v;
        }
    }
    // Compute data size and allocate tensor.
    const int data_size_in_bytes
        = kSN3TypeToSize.at(sn3_file.type) * std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    sn3_file.tensor = infini_train::Tensor(dims, kSN3TypeToDataType.at(sn3_file.type));
    // Load raw payload into tensor memory.
    ifs.read(reinterpret_cast<char *>(sn3_file.tensor.DataPtr()), data_size_in_bytes);
    return sn3_file;
}
} // namespace

MNISTDataset::MNISTDataset(const std::string &dataset, bool train)
    : image_file_(
        ReadSN3PascalVincentFile(std::format("{}/{}-images-idx3-ubyte", dataset, train ? kTrainPrefix : kTestPrefix))),
      label_file_(ReadSN3PascalVincentFile(
          std::format("{}/{}-labels-idx1-ubyte", dataset, train ? kTrainPrefix : kTestPrefix))),
      image_dims_(image_file_.dims.begin() + 1, image_file_.dims.end()),
      label_dims_(label_file_.dims.begin() + 1, label_file_.dims.end()) {

    // Sanity checks.
    CHECK_EQ(image_file_.dims[0], label_file_.dims[0]);
    CHECK_EQ(static_cast<int>(image_file_.tensor.Dtype()), static_cast<int>(DataType::kUINT8));
    const auto &image_dims = image_file_.tensor.Dims();
    CHECK_EQ(image_dims.size(), 3);
    CHECK_EQ(image_dims[1], 28);
    CHECK_EQ(image_dims[2], 28);

    // Convert images to float32 and normalize to [0, 1].
    const auto bs = image_dims[0];
    infini_train::Tensor transposed_tensor(image_dims, DataType::kFLOAT32);
    for (int idx = 0; idx < bs; ++idx) {
        const auto *image_data = reinterpret_cast<uint8_t *>(image_file_.tensor.DataPtr()) + idx * 28 * 28;
        auto *transposed_data = static_cast<float *>(transposed_tensor.DataPtr()) + idx * 28 * 28;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) { transposed_data[i * 28 + j] = image_data[i * 28 + j] / 255.0f; }
        }
    }
    // Replace raw uint8 image tensor.
    image_file_.tensor = std::move(transposed_tensor);
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
MNISTDataset::operator[](size_t idx) const {
    // Bounds check.
    CHECK_LT(idx, image_file_.dims[0]);
    const size_t image_size_in_bytes = BytesPerSample(image_file_);
    const size_t label_size_in_bytes = BytesPerSample(label_file_);

    // Build per-sample tensor views by offset.
    return {std::make_shared<infini_train::Tensor>(image_file_.tensor, idx * image_size_in_bytes, image_dims_),
            std::make_shared<infini_train::Tensor>(label_file_.tensor, idx * label_size_in_bytes, label_dims_)};
}

// Number of samples.
size_t MNISTDataset::Size() const { return image_file_.dims[0]; }
