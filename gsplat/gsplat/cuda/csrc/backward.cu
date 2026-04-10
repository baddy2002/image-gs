#include "backward.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile){
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile){
    val = cg::reduce(tile, val, cg::plus<float>());
}

__global__ void nd_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ beta,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_beta
) {

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float rgbs_batch[BLOCK_SIZE][MAX_CHANNELS];
    __shared__ float beta_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float *v_out = &(v_output[channels * pix_id]);
    
    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing

    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from front to back
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussians_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            xy_batch[tr] = {xy.x, xy.y};
            conic_batch[tr] = conics[g_id];
            beta_batch[tr] = beta[g_id];
            for (int c = 0; c < channels; ++c) 
                rgbs_batch[tr][c] = rgbs[channels * g_id + c];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && inside; ++t) {

            float3 conic = conic_batch[t];
            float2 xy = xy_batch[t];
            float b = beta_batch[t];
            float2 delta = {xy.x - px, xy.y - py};
            float d_squared =  (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            
            if (d_squared > 1.0f || d_squared < 0.f || isnan(d_squared) || isinf(d_squared)) {
                continue;
            }

            float  v_rgb_local[MAX_CHANNELS] = {0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_beta_local = 0.f;
            
            const float base = max(1e-4f, 1.0f - d_squared);
            const float alpha = powf(base, b);

            // gradiente rispetto al colore
            for (int c = 0; c < channels; ++c)
                v_rgb_local[c] = alpha * v_out[c];

            const float* rgb = rgbs_batch[t];

            /* update v_sigma for this gaussian
            float v_sigma = 0.f;
            for (int c = 0; c < channels; ++c)
                v_sigma += rgb[c] * v_out[c];
            v_sigma *= -d;
            */
            //gradiente rispetto ad alpha
            float v_alpha = 0.f;
            for(int c = 0; c < channels; ++c)
                v_alpha += rgb[c] * v_out[c];
            
            //  Gradiente rispetto a BETA 
            v_beta_local = v_alpha * alpha * logf(max(base, 1e-6f));

            // Gradiente rispetto alla DISTANZA (d_squared)
            const float v_dist = v_alpha * (-b) * powf(base, max(b - 1.0f, 1e-6f));

            // gradiente rispetto alla conica (ABC)
            v_conic_local = { 
                            v_dist * delta.x * delta.x, 
                            v_dist * delta.x * delta.y, 
                            v_dist * delta.y * delta.y};

            // gradiente rispetto xy (centroide)
            v_xy_local = {v_dist * (2.0f * conic.x * delta.x + conic.y * delta.y), 
                            v_dist * (2.0f * conic.z * delta.y + conic.y * delta.x)};
            
            // sum across the warp
            for (int c = 0; c < channels; ++c)
                warpSum(v_rgb_local[c], warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_beta_local, warp);

            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                for (int c = 0; c < channels; ++c)
                    atomicAdd(v_rgb_ptr + channels * g + c, v_rgb_local[c]);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);

                float* v_beta_ptr = (float*)(v_beta);
                atomicAdd(v_beta_ptr + g, v_beta_local);
            }
        }
    }
}

__global__ void nd_rasterize_backward_topk_norm_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t* __restrict__ gaussians_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ beta,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_beta,
    int* __restrict__ pixel_topk
) {
    
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);
    if (!inside) return;

    // df/d_out for this pixel
    const float* v_out = &(v_output[channels * pix_id]);
    // topk gs id for this pixel
    const int* topk = &pixel_topk[pix_id * TOP_K];
    
    // compute the normalization factor
    float alpha_local[TOP_K] = {0.0f};
    float base_local[TOP_K] = {0.0f};
    float denom = EPS;
    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;

        float3 conic = conics[g_id];
        float2 xy = xys[g_id];
        float b = beta[g_id];

        float2 delta = {xy.x - px, xy.y - py};
        float d_squared = (
                        conic.x * delta.x * delta.x +
                        conic.z * delta.y * delta.y +
                        conic.y * delta.x * delta.y);
        const float base = max(EPS, 1.0f - d_squared);
        float alpha = powf(base, b);
        alpha_local[k] = alpha;
        base_local[k] = base;
        denom += alpha;
    }
    
    //DERIVATA DELLA NORMALIZZAZIONE ---
    float v_alpha_local[TOP_K] = {0.f};

    
    // Ogni alpha_k influenza tutti i pesi normalizzati
    // FORMULA STABILE: 
    // trasformiamo il calcolo della regola del quozionte f'(x)g(x)-f(x)g'(x) / g(x)^2: 
    // v_norm_weight * 1/D - (alpha * v_norm_weight / D^2) - sum_{l!=k} (alpha_l * v_norm_weight / D^2) 
    //eliminando il denominatore comune D^2 per essere più stabili numericamente:
    // 1/D * [v_norm_weight - sum_{l} (v_norm_weight * alpha_l / D )] = 
    // 1/D * [v_norm_weight - sum_{l} (v_norm_weight * w_l )]
    float weighted_v_sum = 0.f;
    for (int j = 0; j < TOP_K; ++j) {
        if (topk[j] >= 0) {
            float w_j = alpha_local[j] / denom;    //calcola il peso normalizzato di questa gaussiana
            // Calcola quanto i colori degli ALTRI influenzano questo gradiente
            float v_nw_j = 0.f;
            const float* rgb_j = &rgbs[channels * topk[j]];
            for (int c = 0; c < channels; ++c) v_nw_j += rgb_j[c] * v_out[c];
            weighted_v_sum += v_nw_j * w_j; //la media pesata dei gradienti delle gaussiane su quel pixel
        }
    }
    
    // compute each gaussian's contribution to the gradient
    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;

        float alpha = alpha_local[k];
        float norm_weight = alpha / max(denom, 1e-7f);

        float v_rgb_local[MAX_CHANNELS] = {0.f};

        // update v_rgb for this gaussian
        
        const float* rgb = &rgbs[channels * g_id];
        float* v_rgb_ptr = (float*)(v_rgb);
        for (int c = 0; c < channels; ++c)
            atomicAdd(v_rgb_ptr + channels * g_id + c, norm_weight * v_out[c]);
        
        // Quanto la Loss cambia rispetto al peso normalizzato di questa gaussiana
        float v_norm_weight = 0.f;
        for (int c = 0; c < channels; ++c)
            v_norm_weight += rgb[c] * v_out[c];

        /*
        // Formula derivata Softmax: dL/d_alpha
        // Ogni alpha_k influenza tutti i pesi normalizzati
        // usiamo regola del quozionte f'(x)g(x)-f(x)g'(x) / g(x)^2
        // solo se l==k: v_norm_weight * 1/D
        float term_denom = v_norm_weight / denom;
        // termine condiviso (-alpha )/D^2 per ogni alpha_k
        float term_shared = (v_norm_weight * (-alpha )) / (denom * denom);
        for (int l = 0; l < TOP_K; ++l) {
            if (l == k) {
                v_alpha_local[l] += term_denom + term_shared;
            } 
            else {
                v_alpha_local[l] += term_shared;
            }
        }        
        */
        v_alpha_local[k] = (v_norm_weight - weighted_v_sum) / (denom);
    }
    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;

        float v_alpha = v_alpha_local[k];
        const float* rgb = &rgbs[channels * g_id];
        float b = beta[g_id];
        float base = base_local[k];
        float alpha = alpha_local[k];

        // gradiente rispetto a beta
        float v_beta_local = v_alpha * alpha * logf(max(base, 1e-4f));

        // Gradiente rispetto alla DISTANZA (d_squared)
        float v_dist = v_alpha * (-b) * powf(base, max(b - 1.0f, 1e-6f));

        float3 conic = conics[g_id];
        float2 xy = xys[g_id];
        float2 delta = {xy.x - px, xy.y - py};
        
        // gradiente rispetto alla conica
        float3 v_conic_local = { 
                            v_dist * delta.x * delta.x, 
                            v_dist * delta.x * delta.y, 
                            v_dist * delta.y * delta.y};
        
        // gradiente rispetto alla posizione (centroide)
        float2 v_xy_local = {
                    v_dist * (2.0f * conic.x * delta.x + conic.y * delta.y), 
                    v_dist * (2.0f * conic.z * delta.y + conic.y * delta.x)};
        

        float* v_conic_ptr = (float*)(v_conic);
        atomicAdd(v_conic_ptr + 3*g_id + 0, v_conic_local.x);
        atomicAdd(v_conic_ptr + 3*g_id + 1, v_conic_local.y);
        atomicAdd(v_conic_ptr + 3*g_id + 2, v_conic_local.z);
        
        float* v_xy_ptr = (float*)(v_xy);
        atomicAdd(v_xy_ptr + 2*g_id + 0, v_xy_local.x);
        atomicAdd(v_xy_ptr + 2*g_id + 1, v_xy_local.y);

        float* v_beta_ptr = (float*)(v_beta);
        atomicAdd(v_beta_ptr + g_id, v_beta_local);
    }
}

__global__ void nd_rasterize_backward_no_tiles_kernel(
    const dim3 img_size,
    const unsigned channels,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float* __restrict__ rgbs,
    const float* __restrict__ beta,
    const float* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float* __restrict__ v_rgb,
    float* __restrict__ v_beta,
    int* __restrict__ pixel_topk
) {
    
    auto block = cg::this_thread_block();
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);
    if (!inside) return;

    // df/d_out for this pixel
    const float* v_out = &(v_output[channels * pix_id]);
    // topk gs id for this pixel
    const int* topk = &pixel_topk[pix_id * TOP_K];
    
    // compute the normalization factor

    float alpha_local[TOP_K] = {0.0f};
    float base_local[TOP_K] = {0.0f};

    float denom = EPS_no_tiles;
    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;

        float3 conic = conics[g_id];
        float2 xy = xys[g_id];
        float b = beta[g_id];

        float2 delta = {xy.x - px, xy.y - py};
        float d_squared =  (conic.x * delta.x * delta.x +
                            conic.z * delta.y * delta.y +
                            conic.y * delta.x * delta.y);
        if (d_squared > 1.0f || d_squared < 0.f || isnan(d_squared) || isinf(d_squared)) {
            continue;
        }
        
        const float base = max(1e-4f, 1.0f - d_squared);
        float alpha = powf(base, b);
        denom += alpha;
        alpha_local[k] = alpha;
        base_local[k] = base;
    }
    
    //DERIVATA DELLA NORMALIZZAZIONE ---
    float v_alpha_local[TOP_K] = {0.f};

    // compute each gaussian's contribution to the gradient
    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;

        float alpha = alpha_local[k];        
        float norm_weigh = alpha / denom;

        // gradiente rispetto al colore 
        const float* rgb = &rgbs[channels * g_id];
        float* v_rgb_ptr = (float*)(v_rgb);
        for (int c = 0; c < channels; ++c)
            atomicAdd(v_rgb_ptr + channels * g_id + c, norm_weigh * v_out[c]);
        
        // Quanto la Loss cambia rispetto al peso normalizzato di questa gaussiana
        float v_norm_weight = 0.f;
        for (int c = 0; c < channels; ++c)
            v_norm_weight += rgb[c] * v_out[c];

        // Formula derivata Softmax: dL/d_alpha
        // Ogni alpha_k influenza tutti i pesi normalizzati
        float term_denom = v_norm_weight / denom;
        float term_shared = - (alpha * v_norm_weight) / (denom * denom);
        for (int l = 0; l < TOP_K; ++l) {
            if (l == k) {
                v_alpha_local[l] += term_denom + term_shared;
            } 
            else {
                v_alpha_local[l] += term_shared;
            }
        }        
    }

    for (int k = 0; k < TOP_K; ++k) {
        int g_id = topk[k];
        if (g_id < 0) continue;

        const float* rgb = &rgbs[channels * g_id];
        float3 conic = conics[g_id];
        float2 xy = xys[g_id];
        float b = beta[g_id];
        float2 delta = {xy.x - px, xy.y - py};
        float base = base_local[k];
        float alpha = alpha_local[k];

        float v_alpha = v_alpha_local[k];        

        // gradiente rispetto a beta
        float v_beta_local = v_alpha * alpha * logf(max(base, 1e-6f));

        // Gradiente rispetto alla DISTANZA (d_squared)
        float v_dist = v_alpha * (-b) * powf(base, max(b - 1.0f, 1e-6f));

        // gradiente per conica (ABC)
        float3 v_conic_local = {
                        v_dist * delta.x * delta.x, 
                        v_dist * delta.x * delta.y, 
                        v_dist * delta.y * delta.y};
        
        // update v_xy for this gaussian
        float2 v_xy_local = {
                    v_dist * (2.0f * conic.x * delta.x + conic.y * delta.y), 
                    v_dist * (2.0f * conic.z * delta.y + conic.y * delta.x)};
         

        float* v_conic_ptr = (float*)(v_conic);
        atomicAdd(v_conic_ptr + 3*g_id + 0, v_conic_local.x);
        atomicAdd(v_conic_ptr + 3*g_id + 1, v_conic_local.y);
        atomicAdd(v_conic_ptr + 3*g_id + 2, v_conic_local.z);
        
        float* v_xy_ptr = (float*)(v_xy);
        atomicAdd(v_xy_ptr + 2*g_id + 0, v_xy_local.x);
        atomicAdd(v_xy_ptr + 2*g_id + 1, v_xy_local.y);

        float* v_beta_ptr = (float*)(v_beta);
        atomicAdd(v_beta_ptr + g_id, v_beta_local);
    }
}

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float3* __restrict__ conics,
    const float3* __restrict__ rgbs,
    const float* __restrict__ beta,
    const float3* __restrict__ v_output,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_beta
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float3 rgbs_batch[BLOCK_SIZE];
    __shared__ float beta_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    bool valid = inside;
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from front to back
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            xy_batch[tr] = {xy.x, xy.y};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
            beta_batch[tr] = beta[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; t < batch_size; ++t) {

            float3 conic = conic_batch[t];
            float2 xy = xy_batch[t];
            float b = beta_batch[t];
            float2 delta = {xy.x - px, xy.y - py};
            
            /*
            float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            */
           const float d_squared = (conic.x * delta.x * delta.x +
                                conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            //float d = __expf(-sigma);
            if (d_squared >= 1.0f || d_squared < 0.f || isnan(d_squared) || isinf(d_squared)) {
                valid = 0;
            }
            
            //variabili locali per gradienti di questo pixel
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_beta_local = 0.f;
            
            if (valid) {
                const float base = max(1e-4f, 1.0f - d_squared);
                const float alpha = powf(base, b);
                
                // gradiente rispetto al colore
                v_rgb_local = {alpha * v_out.x, alpha * v_out.y, alpha * v_out.z};

                const float3 rgb = rgbs_batch[t];
                
                //Gradiente rispetto ad alpha (opacità)
                //poichè l'opacità influisce su tutti i canali RGB somma i contributi,
                //il gradiente di alpha deve infatti avere un unico valore essendo alpha uno scalare
                // v_out*color, è un risultato intermedio per i gradienti di beta e distanza 
                const float v_alpha = (
                    rgb.x * v_out.x + 
                    rgb.y * v_out.y + 
                    rgb.z * v_out.z
                );

                //  Gradiente rispetto a BETA 
                // dL/dbeta = v_alpha * dalpha/dbeta = v_alpha * alpha * ln(base)
                // Usiamo max(base, 1e-6) per evitare che il logaritmo esploda a -inf
                v_beta_local = v_alpha * alpha * logf(max(base, 1e-6f));

                // Gradiente rispetto alla DISTANZA (d_squared)
                // controlliamo esponente non sia sotto lo 0 per evitare gradienti esplosivi quando la distanza è molto piccola
                const float v_dist = v_alpha * (-b) * powf(base, max(b - 1.0f, 1e-6f));

                // update v_conic for this gaussian
                v_conic_local = {   v_dist * delta.x * delta.x,          // dL/dA
                                    v_dist * delta.x * delta.y,    // dL/dB
                                    v_dist * delta.y * delta.y};         // dL/dC
                //  Gradiente rispetto a XY (Centro della gaussiana) ---
                // dL/dx = v_dist * ddist/dx. Poiché dx = (x - px), ddist/dx = 2A*dx + B*dy
                v_xy_local = {v_dist * (2.0f * conic.x * delta.x + conic.y * delta.y),
                              v_dist * (2.0f * conic.z * delta.y + conic.y * delta.x)};
            }
            
            // sum across the warp
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_beta_local, warp);

            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3*g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3*g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3*g + 2, v_rgb_local.z);
                
                float* v_conic_ptr = (float*)(v_conic);
                atomicAdd(v_conic_ptr + 3*g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3*g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3*g + 2, v_conic_local.z);
                
                float* v_xy_ptr = (float*)(v_xy);
                atomicAdd(v_xy_ptr + 2*g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2*g + 1, v_xy_local.y);

                float* v_beta_ptr = v_beta;
                atomicAdd(v_beta_ptr + g, v_beta_local);
            }
        }
    }
}


// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float* __restrict__ v_cov3d,
    float3& __restrict__ v_scale,
    float4& __restrict__ v_quat
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    glm::mat3 v_V = glm::mat3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    glm::mat3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::mat3 v_M = 2.f * v_V * M;
    // glm::mat3 v_S = glm::transpose(R) * v_M;
    v_scale.x = (float)glm::dot(R[0], v_M[0]);
    v_scale.y = (float)glm::dot(R[1], v_M[1]);
    v_scale.z = (float)glm::dot(R[2], v_M[2]);

    glm::mat3 v_R = v_M * S;
    v_quat = quat_to_rotmat_vjp(quat, v_R);
}
