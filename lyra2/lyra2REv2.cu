extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
}

#include "miner.h"
#include "cuda_helper.h"


static _ALIGN(64) uint64_t *d_hash[MAX_GPUS];
static  uint64_t *d_hash2[MAX_GPUS];

extern void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash);
extern void blake256_cpu_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash);
extern void Keccak256_cpu_hash_32(int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);

extern void keccak256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void keccak256_cpu_init(int thr_id, uint32_t threads);

extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);
extern void skein256_cpu_init(int thr_id, uint32_t threads);

extern void skeinCube256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash);


extern void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, uint32_t tpb);
extern void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t* matrix);

extern void bmw256_cpu_init(int thr_id, uint32_t threads);
extern void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resultnonces, uint32_t target);

extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash);

extern "C" void lyra2v2_hash(void *state, const void *input)
{
	sph_blake256_context      ctx_blake;
	sph_keccak256_context     ctx_keccak;
	sph_skein256_context      ctx_skein;
	sph_bmw256_context        ctx_bmw;
	sph_cubehash256_context   ctx_cube;

	uint32_t hashA[8], hashB[8];

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashB, 32);
	sph_cubehash256_close(&ctx_cube, hashA);


	LYRA2(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 32);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);


	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_lyra2v2(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	uint32_t intensity = 256 * 256 * 4;
	uint32_t tpb = 32;
	//bool mergeblakekeccak = false;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device_map[thr_id]);
	if (!opt_eco_mode)
	{
		if (strstr(props.name, "TITAN X"))
		{
			tpb = 32;
			intensity = 256 * 256 * 32;
		}
		else if (strstr(props.name, "970"))
		{
			tpb = 32;
			intensity = 256 * 256 * 32;
		}
		else if (strstr(props.name, "980 Ti"))
		{
			tpb = 32;
			intensity = 256 * 256 * 32;
		}
		else if (strstr(props.name, "980"))
		{
			tpb = 32;
			intensity = 256 * 256 * 32;
		}
		else if (strstr(props.name, "750 Ti"))
		{
			intensity = 256 * 256 * 4;
			tpb = 32;
			//mergeblakekeccak = true;
		}
		else if (strstr(props.name, "750"))
		{
			intensity = 256 * 256 * 4;
			tpb = 32;
			//mergeblakekeccak = true;
		}
		else if (strstr(props.name, "960"))
		{
			tpb = 32;
			intensity = 256 * 256 * 8;
		}
		else if (strstr(props.name, "950"))
		{
			intensity = 256 * 256 * 8;
			tpb = 32;
		}
	}
	else
	{
		if (strstr(props.name, "TITAN X"))
		{
			tpb = 32;
			intensity = 256 * 256 * 4;
		}
		else if (strstr(props.name, "970"))
		{
			tpb = 32;
			intensity = 256 * 256 * 4;
		}
		else if (strstr(props.name, "980 Ti"))
		{
			tpb = 32;
			intensity = 256 * 256 * 4;
		}
		else if (strstr(props.name, "980"))
		{
			tpb = 32;
			intensity = 256 * 256 * 4;
		}
		else if (strstr(props.name, "750 Ti"))
		{
			intensity = 256 * 256 / 2;
			tpb = 32;
			//mergeblakekeccak = true;
		}
		else if (strstr(props.name, "750"))
		{
			intensity = 256 * 256 / 2;
			tpb = 32;
			//mergeblakekeccak = true;
		}
		else if (strstr(props.name, "960"))
		{
			tpb = 32;
			intensity = 256 * 256 * 1;
		}
		else if (strstr(props.name, "950"))
		{
			intensity = 256 * 256 * 1;
			tpb = 32;
		}
	}
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, intensity);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x00ff;
	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (!opt_cpumining) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		//keccak256_cpu_init(thr_id,throughput);
		skein256_cpu_init(thr_id, throughput);
		bmw256_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash2[thr_id], 4 * sizeof(uint64_t) * 4 * throughput + 128));
		d_hash2[thr_id] = (uint64_t*)(((uint64_t)d_hash2[thr_id] + 127)&~127);
		lyra2v2_cpu_init(thr_id, throughput, d_hash2[thr_id]);
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint32_t) * throughput+128));
		d_hash[thr_id] = (uint64_t*)(((uint64_t)d_hash[thr_id] + 127)&~127);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	blake256_cpu_setBlock_80(pdata);

	do {
		//applog(LOG_WARNING, "GPU #%d: Loop Start! ptarget = %8x%8x%8x%8x%8x%8x%8x%8x", thr_id, ptarget[7], ptarget[6], ptarget[5], ptarget[4], ptarget[3], ptarget[2], ptarget[1], ptarget[0] );
		uint32_t foundNonce[2] = { 0, 0 };

		//		if (mergeblakekeccak)
		//		{
		blakeKeccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);

		/*		}
		else
		{
		blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
		keccak256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		}
		*/

		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		lyra2v2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], tpb);
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id]);
		bmw256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], foundNonce, ptarget[7]);

		//		foundNonce[0] = 0xffffffff;
		if (foundNonce[0] != 0xffffffff)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t vhash64[8];
			be32enc(&endiandata[19], foundNonce[0]);
			lyra2v2_hash(vhash64, endiandata);
			if (vhash64[7] <= Htarg && fulltest(vhash64, ptarget))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (foundNonce[1] != 0xffffffff)
				{
					pdata[21] = foundNonce[1];
					res++;
					if (opt_benchmark)  applog(LOG_INFO, "GPU #%d Found second nounce %08x", thr_id, foundNonce[1], vhash64[7], Htarg);
				}
				pdata[19] = foundNonce[0];
				if (opt_benchmark) applog(LOG_INFO, "GPU #%d Found nounce % 08x", thr_id, foundNonce[0], vhash64[7], Htarg);
				return res;
			}
			else
			{
				if (vhash64[7] > Htarg) // don't show message if it is equal but fails fulltest
					applog(LOG_WARNING, "GPU #%d: result does not validate on CPU!", thr_id);
			}
		}

		pdata[19] += throughput;

	} while (!scan_abort_flag && !work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));
	
	*hashes_done = pdata[19] - first_nonce;
	return 0;
}
