#include "cudf/cudf_utils.hpp"
#include "../operator/cuda_helper.cuh"
#include "gpu_columns.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

std::vector<std::shared_ptr<GPUColumn>> read_parquet_to_gpu_columns(
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
    
    std::vector<duckdb::shared_ptr<GPUColumn>> result;
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

}
