#include "cudf/cudf_utils.hpp"
#include "../operator/cuda_helper.cuh"
#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_physical_table_scan.hpp"
#include "log/logging.hpp"

namespace duckdb {

std::vector<shared_ptr<GPUColumn>> read_parquet_to_gpu_columns(
    const std::string& file_path,
    const std::vector<std::string>& column_names,
    GPUBufferManager* gpuBufferManager
) {
    auto source = cudf::io::source_info(file_path);
    auto builder = cudf::io::parquet_reader_options::builder(source);
    
    if (!column_names.empty()) {
        builder.columns(column_names);
    }
    
    auto table_with_metadata = cudf::io::read_parquet(builder.build());
    auto table = std::move(table_with_metadata.tbl);
    
    std::vector<shared_ptr<GPUColumn>> result;
    auto table_view = table->view();
    for (size_t i = 0; i < table->num_columns(); i++) {
        auto cudf_col = table_view.column(i);
        auto gpu_col = make_shared_ptr<GPUColumn>();
        
        auto cudf_col_owned = std::make_unique<cudf::column>(cudf_col);
        gpu_col->setFromCudfColumn(*cudf_col_owned, false, nullptr, 0, gpuBufferManager);
        
        result.push_back(gpu_col);
    }
    return result;
}

void cache_parquet_columns(std::vector<shared_ptr<GPUColumn>>& columns, GPUBufferManager* gpuBufferManager) 
{
    for (auto& column : columns) {
        auto& data_wrapper = column->data_wrapper;
        size_t num_rows = column->column_length;
        SIRIUS_LOG_DEBUG("Caching parquet column with {} rows", num_rows);
        uint8_t* cached_data = gpuBufferManager->customCudaMalloc<uint8_t>(
            data_wrapper.num_bytes, 0, 1
        );
        cudaMemcpy(cached_data, data_wrapper.data, data_wrapper.num_bytes, cudaMemcpyDeviceToDevice);

        cudf::bitmask_type* cached_mask = nullptr;
        if (data_wrapper.validity_mask != nullptr) {
            cached_mask = gpuBufferManager->customCudaMalloc<cudf::bitmask_type>(
                data_wrapper.mask_bytes / sizeof(cudf::bitmask_type), 0, 1
            );
            cudaMemcpy(cached_mask, data_wrapper.validity_mask,
                      data_wrapper.mask_bytes, cudaMemcpyDeviceToDevice);
        }

        uint64_t* cached_offset = nullptr;
        if (data_wrapper.is_string_data && data_wrapper.offset != nullptr) {
            cached_offset = gpuBufferManager->customCudaMalloc<uint64_t>(num_rows + 1, 0, 1);
            cudaMemcpy(cached_offset, data_wrapper.offset,
                      sizeof(uint64_t) * (num_rows + 1), cudaMemcpyDeviceToDevice);
        }

        data_wrapper.data = cached_data;
        data_wrapper.validity_mask = cached_mask;
        data_wrapper.offset = cached_offset;
    }
    cudaDeviceSynchronize();
}

}
