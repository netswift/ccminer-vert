/**
 * Blake-256 Decred 180-Bytes input Cuda Kernel (Tested on SM 5/5.2)
 * SP-MOD release #4
 * Tanguy Pruvot - Feb 2016
 * SP (amigaguru@gmail.com) march 2016.
 */

#include <stdint.h>
#include <memory.h>


#include <miner.h>

extern "C" {
#include <sph/sph_blake.h>
}

/* threads per block */
#define TPB 768
#define NONCES_PER_THREAD 1024

/* hash by cpu with blake 256 */
extern "C" void decred_hash(void *output, const void *input)
{
	sph_blake256_context ctx;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 180);
	sph_blake256_close(&ctx, output);
}


#include <cuda_helper.h>

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

__constant__ uint32_t _ALIGN(16) d_data[27];
__constant__ uint32_t _ALIGN(16) pre[220];


/* 8 adapters max */
 uint32_t *d_resNonce[MAX_GPUS];
 uint32_t *h_resNonce[MAX_GPUS];

/* max count of found nonces in one call */
#define NBN 2
#if NBN > 1
// uint32_t extra_results[MAX_GPUS][NBN] = { UINT32_MAX };
#endif

#define ROTR32_c(x, n)	__funnelshift_r( (x), (x), (n) )	//(((x) >> (n)) | ((x) << (32 - (n))))

/* ############################################################################################################################### */

 __device__ __forceinline__ uint32_t SWAPWORDS(uint32_t value)
 {
	 ushort2 temp;
	 asm("mov.b32 {%0, %1}, %2; ": "=h"(temp.x), "=h"(temp.y) : "r"(value));
	 asm("mov.b32 %0, {%1, %2}; ": "=r"(value) : "h"(temp.y), "h"(temp.x));
	 return value;
 }

#define RSPRECHOST(x,y) { \
	prehost[i++] =(m[x] ^ u256[y]) ; \
	prehost[i++] =(m[y] ^ u256[x]); \
  }

 // __byte_perm(v[d] ^ v[a], 0, 0x1032); \
 
#define GSPREC(a,b,c,d,x,y) { \
	v[a] += v[b]+(m[x] ^ c_u256[y]) ; \
	v[d] = __byte_perm(v[d] ^ v[a], 0, 0x1032);\
	v[c] += v[d]; \
	v[b] = ROTR32_c(v[b] ^ v[c], 12); \
	v[a] +=  v[b] +(m[y] ^ c_u256[x]); \
	v[d] = __byte_perm(v[d] ^ v[a], 0, 0x0321); \
	v[c] += v[d]; \
	v[b] = ROTR32_c(v[b] ^ v[c], 7); \
}

#define GSPRECSP(a,b,c,d,x,y) { \
	v[d] = __byte_perm(v[d] ^ v[a], 0, 0x1032); \
	v[c] += v[d]; \
	v[b] = ROTR32_c(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ c_u256[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a], 0, 0x0321); \
	v[c] += v[d]; \
	v[b] = ROTR32_c(v[b] ^ v[c], 7); \
  }

#define GSPRECHOST(a, b, c, d, x, y) {\
	v[a] += (m[x] ^ u256[y]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a],16); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ u256[x]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
	}

#define GSPREC_SP(a,b,c,d,x,y) { \
	v[a] += (pre[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a], 0, 0x1032); \
	v[c] += v[d]; \
	v[b] = ROTR32_c(v[b] ^ v[c], 12); \
	v[a] += (pre[y]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a], 0, 0x0321); \
	v[c] += v[d]; \
	v[b] = ROTR32_c(v[b] ^ v[c], 7); \
   }
#define GSPRECHOST(a, b, c, d, x, y) {\
	v[a] += (m[x] ^ u256[y]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a],16); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ u256[x]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
 	}


/* ############################################################################################################################### */


__global__ __launch_bounds__(TPB,1)
void blake256_gpu_hash_nonce(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce)
{
	const uint32_t T0 = 180 * 8;
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
	//	const uint32_t nonce = startNonce + thread;

		const uint32_t numberofthreads = blockDim.x*gridDim.x;
		const uint32_t maxnonce = startNonce + thread + numberofthreads*NONCES_PER_THREAD - 1;
		const uint32_t threadindex = blockIdx.x*blockDim.x + threadIdx.x;
		#pragma unroll
		for (uint32_t nonce = startNonce + threadindex; nonce <= maxnonce; nonce += numberofthreads)
		{

			uint32_t v[16];

#pragma unroll 8
			for (uint32_t i = 0; i < 7; i++)
				v[i] = d_data[i];
			uint32_t backup = d_data[7];
			uint32_t m[16];

			m[0] = d_data[8];
			m[1] = d_data[9];
			m[2] = d_data[10];
			m[3] = nonce;
			v[7] = d_data[11];
#pragma unroll
			for (uint32_t i = 4; i < 13; i++)
			{
				m[i] = d_data[i + 8];
			}

			const uint32_t c_u256[16] = {
				0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
				0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
				0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
				0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
			};
			v[13] = c_u256[5] ^ T0;
			v[8] = d_data[21];//c_u256[0];
			v[12] = d_data[22];//c_u256[4] ^ T0;
			v[10] = d_data[23];
			v[14] = d_data[24];
			v[11] = d_data[25]; //c_u256[3];
			v[15] = d_data[26];//c_u256[7];
			v[9] = c_u256[1];


			m[21 - 8] = 0x80000001;
			m[22 - 8] = 0;
			m[23 - 8] = 0x000005a0;


			// round 1
			//		GSPREC(0, 4, 0x8, 0xC, 0, 1);
			GSPRECSP(1, 5, 0x9, 0xD, 2, 3);
			//			GSPREC_SP(1, 5, 0x9, 0xD, 2, 3);

			//		GSPREC(2, 6, 0xA, 0xE, 4, 5);
			//	GSPREC(3, 7, 0xB, 0xF, 6, 7);

			//		GSPREC(0, 5, 0xA, 0xF, 8, 9);
			v[0] += v[5];
			v[0xf] = __byte_perm(v[0xf] ^ v[0], 0, 0x1032);
			v[0xa] += v[0xf];
			v[5] = SPH_ROTR32(v[5] ^ v[0xa], 12);
			v[0] += (pre[9]) + v[5];
			v[0xf] = __byte_perm(v[0xf] ^ v[0], 0, 0x0321);
			v[0xa] += v[0xf];
			v[5] = SPH_ROTR32(v[5] ^ v[0xa], 7);

			GSPREC(1, 6, 0xB, 0xC, 10, 11);

			//		GSPREC(2, 7, 0x8, 0xD, 12, 13);
			v[0xD] = __byte_perm(v[0xD] ^ v[2], 0, 0x1032);
			v[0x8] += v[0xD];
			v[7] = SPH_ROTR32(v[7] ^ v[8], 12);
			v[2] += (pre[13]) + v[7];
			v[0xD] = __byte_perm(v[0xD] ^ v[2], 0, 0x0321);
			v[0x8] += v[0xD];
			v[7] = SPH_ROTR32(v[7] ^ v[8], 7);

			//	GSPREC(3, 4, 0x9, 0xE, 14, 15);
			v[0x9] += v[0xe];
			v[4] = SPH_ROTR32(v[4] ^ v[9], 12);
			v[3] += (pre[15]) + v[4];
			v[0xe] = __byte_perm(v[0xe] ^ v[3], 0, 0x0321);
			v[0x9] += v[0xe];
			v[4] = SPH_ROTR32(v[4] ^ v[0x9], 7);

			GSPREC_SP(0, 4, 0x8, 0xC, 16, 17);
			GSPREC_SP(1, 5, 0x9, 0xD, 18, 19);
			GSPREC_SP(2, 6, 0xA, 0xE, 20, 21);
			GSPREC_SP(3, 7, 0xB, 0xF, 22, 23);
			GSPREC_SP(0, 5, 0xA, 0xF, 24, 25);
			GSPREC_SP(1, 6, 0xB, 0xC, 26, 27);
			GSPREC_SP(2, 7, 0x8, 0xD, 28, 29);
			//			GSPREC_SP(3, 4, 0x9, 0xE, 30, 31);
			GSPREC(3, 4, 0x9, 0xE, 5, 3);


			// round 3
			GSPREC_SP(0, 4, 0x8, 0xC, 32, 33);
			GSPREC_SP(1, 5, 0x9, 0xD, 34, 35);
			GSPREC_SP(2, 6, 0xA, 0xE, 36, 37);
			GSPREC_SP(3, 7, 0xB, 0xF, 38, 39);
			GSPREC_SP(0, 5, 0xA, 0xF, 40, 41);
			//			GSPREC_SP(1, 6, 0xB, 0xC, 42, 43);
			GSPREC(1, 6, 0xB, 0xC, 3, 6);
			GSPREC_SP(2, 7, 0x8, 0xD, 44, 45);
			GSPREC_SP(3, 4, 0x9, 0xE, 46, 47);

			// round 4
			GSPREC_SP(0, 4, 0x8, 0xC, 48, 49);
			//			GSPREC_SP(1, 5, 0x9, 0xD, 50, 51);
			GSPREC(1, 5, 0x9, 0xD, 3, 1);
			GSPREC_SP(2, 6, 0xA, 0xE, 52, 53);
			GSPREC_SP(3, 7, 0xB, 0xF, 54, 55);
			GSPREC_SP(0, 5, 0xA, 0xF, 56, 57);
			GSPREC_SP(1, 6, 0xB, 0xC, 58, 59);
			GSPREC_SP(2, 7, 0x8, 0xD, 60, 61);
			GSPREC_SP(3, 4, 0x9, 0xE, 62, 63);
			// round 5
			GSPREC_SP(0, 4, 0x8, 0xC, 64, 65);
			GSPREC_SP(1, 5, 0x9, 0xD, 66, 67);
			GSPREC_SP(2, 6, 0xA, 0xE, 68, 69);
			GSPREC_SP(3, 7, 0xB, 0xF, 70, 71);
			GSPREC_SP(0, 5, 0xA, 0xF, 72, 73);
			GSPREC_SP(1, 6, 0xB, 0xC, 74, 75);
			GSPREC_SP(2, 7, 0x8, 0xD, 76, 77);
			//			GSPREC_SP(3, 4, 0x9, 0xE, 78, 79);
			GSPREC(3, 4, 0x9, 0xE, 3, 13);
			// round 6
			GSPREC_SP(0, 4, 0x8, 0xC, 80, 81);
			GSPREC_SP(1, 5, 0x9, 0xD, 82, 83);
			GSPREC_SP(2, 6, 0xA, 0xE, 84, 85);
			//		GSPREC_SP(3, 7, 0xB, 0xF, 86, 87);
			GSPREC(3, 7, 0xB, 0xF, 8, 3);
			GSPREC_SP(0, 5, 0xA, 0xF, 88, 89);
			GSPREC_SP(1, 6, 0xB, 0xC, 90, 91);
			GSPREC_SP(2, 7, 0x8, 0xD, 92, 93);
			GSPREC_SP(3, 4, 0x9, 0xE, 94, 95);
			// round 7
			GSPREC_SP(0, 4, 0x8, 0xC, 96, 97);
			GSPREC_SP(1, 5, 0x9, 0xD, 98, 99);
			GSPREC_SP(2, 6, 0xA, 0xE, 100, 101);
			GSPREC_SP(3, 7, 0xB, 0xF, 102, 103);
			GSPREC_SP(0, 5, 0xA, 0xF, 104, 105);
			//			GSPREC_SP(1, 6, 0xB, 0xC, 106, 107);
			GSPREC(1, 6, 0xB, 0xC, 6, 3);
			GSPREC_SP(2, 7, 0x8, 0xD, 108, 109);
			GSPREC_SP(3, 4, 0x9, 0xE, 110, 111);
			// round 8
			GSPREC_SP(0, 4, 0x8, 0xC, 112, 113);
			GSPREC_SP(1, 5, 0x9, 0xD, 114, 115);
			GSPREC_SP(2, 6, 0xA, 0xE, 116, 117);
			GSPREC(3, 7, 0xB, 0xF, 3, 9);
			//			GSPREC_SP(3, 7, 0xB, 0xF, 118, 119);
			GSPREC_SP(0, 5, 0xA, 0xF, 120, 121);
			GSPREC_SP(1, 6, 0xB, 0xC, 122, 123);
			GSPREC_SP(2, 7, 0x8, 0xD, 124, 125);
			GSPREC_SP(3, 4, 0x9, 0xE, 126, 127);
			// round 9
			GSPREC_SP(0, 4, 0x8, 0xC, 128, 129);
			GSPREC_SP(1, 5, 0x9, 0xD, 130, 131);
			//			GSPREC_SP(2, 6, 0xA, 0xE, 132, 133);
			GSPREC(2, 6, 0xA, 0xE, 11, 3);
			GSPREC_SP(3, 7, 0xB, 0xF, 134, 135);
			GSPREC_SP(0, 5, 0xA, 0xF, 136, 137);
			GSPREC_SP(1, 6, 0xB, 0xC, 138, 139);
			GSPREC_SP(2, 7, 0x8, 0xD, 140, 141);
			GSPREC_SP(3, 4, 0x9, 0xE, 142, 143);
			// round 10

			GSPREC_SP(0, 4, 0x8, 0xC, 144, 145);
			GSPREC_SP(1, 5, 0x9, 0xD, 146, 147);
			GSPREC_SP(2, 6, 0xA, 0xE, 148, 149);
			GSPREC_SP(3, 7, 0xB, 0xF, 150, 151);
			GSPREC_SP(0, 5, 0xA, 0xF, 152, 153);
			GSPREC_SP(1, 6, 0xB, 0xC, 154, 155);
			//			GSPREC_SP(2, 7, 0x8, 0xD, 156, 157);
			GSPREC(2, 7, 0x8, 0xD, 3, 12);
			GSPREC_SP(3, 4, 0x9, 0xE, 158, 159);

			// round 11
			/*			GSPREC(0, 4, 0x8, 0xC, 0, 1);
						GSPREC(1, 5, 0x9, 0xD, 2, 3);
						GSPREC(2, 6, 0xA, 0xE, 4, 5);
						GSPREC(3, 7, 0xB, 0xF, 6, 7);
						GSPREC(0, 5, 0xA, 0xF, 8, 9);
						GSPREC(1, 6, 0xB, 0xC, 10, 11);
						GSPREC(2, 7, 0x8, 0xD, 12, 13);
						GSPREC(3, 4, 0x9, 0xE, 14, 15);
						*/
			GSPREC_SP(0, 4, 0x8, 0xC, 160, 161);
			//			GSPREC_SP(1, 5, 0x9, 0xD, 162, 163);
			GSPREC(1, 5, 0x9, 0xD, 2, 3);
			GSPREC_SP(2, 6, 0xA, 0xE, 164, 165);
			GSPREC_SP(3, 7, 0xB, 0xF, 166, 167);
			GSPREC_SP(0, 5, 0xA, 0xF, 168, 169);
			GSPREC_SP(1, 6, 0xB, 0xC, 170, 171);
			GSPREC_SP(2, 7, 0x8, 0xD, 172, 173);
			GSPREC_SP(3, 4, 0x9, 0xE, 174, 175);


			// round 12
			GSPREC_SP(0, 4, 0x8, 0xC, 176, 177);
			GSPREC_SP(1, 5, 0x9, 0xD, 178, 179);
			GSPREC_SP(2, 6, 0xA, 0xE, 180, 181);
			GSPREC_SP(3, 7, 0xB, 0xF, 182, 183);
			GSPREC_SP(0, 5, 0xA, 0xF, 184, 185);
			GSPREC_SP(1, 6, 0xB, 0xC, 186, 187);
			GSPREC_SP(2, 7, 0x8, 0xD, 188, 189);
			//			GSPREC_SP(3, 4, 0x9, 0xE, 190, 191);
			GSPREC(3, 4, 0x9, 0xE, 5, 3);

			// round 13
			GSPREC_SP(0, 4, 0x8, 0xC, 192, 193);
			GSPREC_SP(1, 5, 0x9, 0xD, 194, 195);
			GSPREC_SP(2, 6, 0xA, 0xE, 196, 197);
			GSPREC_SP(3, 7, 0xB, 0xF, 198, 199);
			GSPREC_SP(0, 5, 0xA, 0xF, 200, 201);
			//			GSPREC_SP(1, 6, 0xB, 0xC, 202, 203);
			GSPREC(1, 6, 0xB, 0xC, 3, 6);
			GSPREC_SP(2, 7, 0x8, 0xD, 204, 205);
			GSPREC_SP(3, 4, 0x9, 0xE, 206, 207);
			// round 14
			GSPREC_SP(0, 4, 0x8, 0xC, 208, 209);
			//			GSPREC_SP(1, 5, 0x9, 0xD, 210, 211);
			GSPREC(1, 5, 0x9, 0xD, 3, 1);
			GSPREC_SP(2, 6, 0xA, 0xE, 212, 213);
			//			GSPREC(3, 7, 0xB, 0xF, 11, 14);

			v[3] += (pre[214]) + v[7];
			v[0xF] = __byte_perm(v[0xF] ^ v[3], 0, 0x1032);
			v[0xB] += v[0xF];
			v[7] = ROTR32_c(v[7] ^ v[0xB], 12);
			v[3] += (pre[215]) + v[7];
			v[0xF] = __byte_perm(v[0xF] ^ v[3], 0, 0x0321);
			v[0xB] += v[0xF];
			v[7] = ROTR32_c(v[7] ^ v[0xB], 7);

			//			GSPREC(0, 5, 0xA, 0xF, 2, 6);
			//#define GSPREC(a,b,c,d,x,y) {
			v[0] += (pre[216]) + v[5];
			v[0xF] = __byte_perm(v[0xF] ^ v[0], 0, 0x1032);
			v[0xA] += v[0xF];
			v[5] = ROTR32_c(v[5] ^ v[0xA], 12);
			v[0] += (pre[217]) + v[5];
			v[0xF] = __byte_perm(v[0xF] ^ v[0], 0, 0x0321);
			v[0xA] += v[0xF];
			//	v[5] = ROTR32_c(v[5] ^ v[0xA], 7);


			//		GSPREC(1, 6, 0xB, 0xC, 5, 10);
			//			GSPREC(2, 7, 0x8, 0xD, 4, 0);
			//#define GSPREC(2,b,c,d,x,y) { 
			v[2] += (m[4] ^ c_u256[0]) + v[7];
			v[0xD] = __byte_perm(v[0xD] ^ v[2], 0, 0x1032);
			v[0x8] += v[0xD];
			v[7] = ROTR32_c(v[7] ^ v[0x8], 12);
			v[2] += (m[0] ^ c_u256[4]) + v[7];
			v[0x8] += __byte_perm(v[0xD] ^ v[2], 0, 0x0321);
			//	v[7] = ;
			//			}
			//GSPREC(3, 4, 0x9, 0xE, 15, 8);

			//	v[7] ^= d_data[7] ^ v[15];

			if (!(ROTR32_c(v[7] ^ v[0x8], 7) ^ backup ^ v[15]))
			{
				/*	v[3] += (m[0xf] ^ c_u256[8]) + v[4];
					v[0xe] = __byte_perm(v[0xe] ^ v[3], 0, 0x1032);
					v[9] += v[0xe]; \
					v[4] = ROTR32(v[4] ^ v[9], 12);
					v[3] += (m[8] ^ c_u256[0xf]) + v[4];
					v[0xe] = __byte_perm(v[0xe] ^ v[3], 0, 0x0321);

					// only compute h6 & 7
					v[6] ^= d_data[6] ^ v[14];


					if (cuda_swab32(v[6]) <= highTarget)
					{
					*/
			/*	if (m[3] < resNonce[0])
				{
					resNonce[1] = resNonce[0];
					resNonce[0] = m[3];
				}
				else
					resNonce[1] = m[3];
				*/

				uint32_t tmp = atomicCAS(resNonce, 0xffffffff, m[3]);
				if (tmp != 0xffffffff)
					resNonce[1] = m[3];
				//}

			}
		}
	}
}

__host__ void decred_cpu_hash_nonce(const int thr_id, const uint32_t threads, const uint32_t startNonce)
{
//	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB*NONCES_PER_THREAD - 1) / TPB / NONCES_PER_THREAD);
	dim3 block(TPB);



	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	cudaMemset(d_resNonce[thr_id], 0xffffffff, NBN*sizeof(uint32_t));
	blake256_gpu_hash_nonce <<<grid, block>>> (threads, startNonce, d_resNonce[thr_id]);
	cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
//	extra_results[thr_id][0] = h_resNonce[thr_id][0];
//	extra_results[thr_id][1] = h_resNonce[thr_id][1];
}

__host__
void decred_midstate_128(uint32_t *output, const uint32_t *input)
{
	sph_blake256_context ctx;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 128);



	memcpy(output, (void*)ctx.H, 32);
}

__host__
void decred_cpu_setBlock_52(uint32_t *penddata, const uint32_t *midstate, const uint32_t *ptarget)
{
	uint32_t _ALIGN(64) data[27];
	uint32_t _ALIGN(64) prehost[250];

//	memcpy(data, midstate, 32);
	// pre swab32
	uint32_t v[16];

	data[0] = midstate[0];
	data[1] = midstate[1];
	data[2] = midstate[2];
	data[3] = midstate[3];
	data[4] = midstate[4];
	data[5] = midstate[5];
	data[6] = midstate[6];
	data[7] = midstate[7];

	for (int i = 0; i<13; i++)
		data[8+i] = swab32(penddata[i]);
//	data[21] = 0x80000001;
//	data[22] = 0;
//	data[23] = 0x000005a0;
	const uint32_t T0 = 180 * 8;


//	for (uint32_t i = 0; i < 8; i++)
//		v[i] = data[i];

	const uint32_t u256[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	v[0] = data[0];
	v[1] = data[1];
	v[2] = data[2];
	v[3] = data[3];
	v[4] = data[4];
	v[5] = data[5];
	v[6] = data[6];
	v[7] = data[7];




	v[8] = u256[0];
	v[9] = u256[1];
	v[10] = u256[2];
	v[11] = u256[3];

	v[12] = u256[4] ^ T0;
	v[13] = u256[5] ^ T0;
	v[14] = u256[6];
	v[15] = u256[7];

	uint32_t m[16];

	m[0] = data[8];
	m[1] = data[9];
	m[2] = data[10];
	m[3] = 0;

	for (uint32_t i = 4; i < 13; i++) {
		m[i] = data[i + 8];
	}

	m[21 - 8] = 0x80000001;
	m[22 - 8] = 0;
	m[23 - 8] = 0x000005a0;

	v[1] += (m[2] ^ u256[3]) + v[5]; 
	GSPRECHOST(0, 4, 0x8, 0xC, 0, 1);
	GSPRECHOST(2, 6, 0xA, 0xE, 4, 5);
	GSPRECHOST(3, 7, 0xB, 0xF, 6, 7);

/*	v[3] += (m[6] ^ u256[7]) + v[7];
	v[0xF] = ROTR32(v[0xF] ^ v[3], 16);
	v[0xB] += v[0xF];
	v[7] = ROTR32(v[7] ^ v[0xB], 12);
	v[3] += (m[7] ^ u256[6]) + v[7];
	v[0xF] = ROTR32(v[0xF] ^ v[3], 8);
	v[0xB] += v[0xF];
	v[7] = ROTR32(v[7] ^ v[0xB], 7);
*/

	v[0] += (m[8] ^ u256[9]);
	v[2] += (m[12] ^ u256[13]) + v[7];
	
	v[3] += (m[14] ^ u256[15]) + v[4];
	v[0xe] = ROTR32(v[0xe] ^ v[3], 16);



	data[0]=v[0];
	data[1] = v[1];
	data[2] = v[2];
	data[3] = v[3];
	data[4] = v[4];
	data[6] = v[6];
//	data[7] = v[7];
	data[11] = v[7];

	data[21] = v[8];
	data[22] = v[0xc];
	data[23] = v[0xa];
	data[24] = v[0xe];
	data[25] = v[0xb];
	data[26] = v[0xf];


	int i = 0;

	RSPRECHOST(0, 1);
	RSPRECHOST(2, 3);
	RSPRECHOST(4, 5);
	RSPRECHOST(6, 7);

	RSPRECHOST(8, 9);
	RSPRECHOST(10, 11);
	RSPRECHOST(12, 13);
	RSPRECHOST(14, 15);
	// round 2
	RSPRECHOST(14, 10);
	RSPRECHOST(4, 8);
	RSPRECHOST(9, 15);
	RSPRECHOST(13, 6);
	RSPRECHOST(1, 12);
	RSPRECHOST(0, 2);
	RSPRECHOST(11, 7);
	RSPRECHOST(5, 3);
	// round 3
	RSPRECHOST(11, 8);
	RSPRECHOST(12, 0);
	RSPRECHOST(5, 2);
	RSPRECHOST(15, 13);
	RSPRECHOST(10, 14);
	RSPRECHOST(3, 6);
	RSPRECHOST(7, 1);
	RSPRECHOST(9, 4);
	// round 4
	RSPRECHOST(7, 9);
	RSPRECHOST(3, 1);
	RSPRECHOST(13, 12);
	RSPRECHOST(11, 14);
	RSPRECHOST(2, 6);
	RSPRECHOST(5, 10);
	RSPRECHOST(4, 0);
	RSPRECHOST(15, 8);
	// round 5
	RSPRECHOST(9, 0);
	RSPRECHOST(5, 7);
	RSPRECHOST(2, 4);
	RSPRECHOST(10, 15);
	RSPRECHOST(14, 1);
	RSPRECHOST(11, 12);
	RSPRECHOST(6, 8);
	RSPRECHOST(3, 13);
	// round 6
	RSPRECHOST(2, 12);
	RSPRECHOST(6, 10);
	RSPRECHOST(0, 11);
	RSPRECHOST(8, 3);
	RSPRECHOST(4, 13);
	RSPRECHOST(7, 5);
	RSPRECHOST(15, 14);
	RSPRECHOST(1, 9);
	// round 7
	RSPRECHOST(12, 5);
	RSPRECHOST(1, 15);
	RSPRECHOST(14, 13);
	RSPRECHOST(4, 10);
	RSPRECHOST(0, 7);
	RSPRECHOST(6, 3);
	RSPRECHOST(9, 2);
	RSPRECHOST(8, 11);
	// round 8
	RSPRECHOST(13, 11);
	RSPRECHOST(7, 14);
	RSPRECHOST(12, 1);
	RSPRECHOST(3, 9);
	RSPRECHOST(5, 0);
	RSPRECHOST(15, 4);
	RSPRECHOST(8, 6);
	RSPRECHOST(2, 10);
	// round 9
	RSPRECHOST(6, 15);
	RSPRECHOST(14, 9);
	RSPRECHOST(11, 3);
	RSPRECHOST(0, 8);
	RSPRECHOST(12, 2);
	RSPRECHOST(13, 7);
	RSPRECHOST(1, 4);
	RSPRECHOST(10, 5);
	// round 10
	RSPRECHOST(10, 2);
	RSPRECHOST(8, 4);
	RSPRECHOST(7, 6);
	RSPRECHOST(1, 5);
	RSPRECHOST(15, 11);
	RSPRECHOST(9, 14);
	RSPRECHOST(3, 12);
	RSPRECHOST(13, 0);
	// round 11
	RSPRECHOST(0, 1);
	RSPRECHOST(2, 3);
	RSPRECHOST(4, 5);
	RSPRECHOST(6, 7);
	RSPRECHOST(8, 9);
	RSPRECHOST(10, 11);
	RSPRECHOST(12, 13);
	RSPRECHOST(14, 15);
	// round 12
	RSPRECHOST(14, 10);
	RSPRECHOST(4, 8);
	RSPRECHOST(9, 15);
	RSPRECHOST(13, 6);
	RSPRECHOST(1, 12);
	RSPRECHOST(0, 2);
	RSPRECHOST(11, 7);
	RSPRECHOST(5, 3);
	// round 13
	RSPRECHOST(11, 8);
	RSPRECHOST(12, 0);
	RSPRECHOST(5, 2);
	RSPRECHOST(15, 13);
	RSPRECHOST(10, 14);
	RSPRECHOST(3, 6);
	RSPRECHOST(7, 1);
	RSPRECHOST(9, 4);
	// round 14
	RSPRECHOST(7, 9);
	RSPRECHOST(3, 1);
	RSPRECHOST(13, 12);

	RSPRECHOST(11, 14);
	RSPRECHOST(2, 6);
	RSPRECHOST(5, 10);
	RSPRECHOST(4, 0);


	(cudaMemcpyToSymbol(d_data, data, 32 + 64 + 4 + 8, 0, cudaMemcpyHostToDevice));
	(cudaMemcpyToSymbol(pre, prehost, 220 * 4, 0, cudaMemcpyHostToDevice));
}

/* ############################################################################################################################### */

bool init[MAX_GPUS] = { 0 };

// nonce position is different in decred
#define DCR_NONCE_OFT32 35

extern "C" int scanhash_decred(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	static uint32_t _ALIGN(64) endiandata[MAX_GPUS][48];
	static uint32_t _ALIGN(64) midstate[MAX_GPUS][8];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *pnonce = &pdata[DCR_NONCE_OFT32];

	const uint32_t first_nonce = *pnonce;
//	uint64_t targetHigh = ((uint64_t*)ptarget)[3];

	int dev_id = device_map[thr_id];
	int intensity = 30;
	if (device_sm[dev_id] < 350) intensity = 22;
	if (device_sm[dev_id] > 500)
	{
		intensity = 30;
	}
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	int rc = 0;

	if (opt_benchmark) 
	{
		//targetHigh = 0x1ULL << 32;
		ptarget[6] = swab32(0xfff);
	}

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		/*	if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
     */
		(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)));
		(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)));
		init[thr_id] = true;
	}

	memcpy(endiandata[thr_id], pdata, 180);
	decred_midstate_128(midstate[thr_id], pdata); //endiandata
	decred_cpu_setBlock_52(&pdata[32], midstate[thr_id], ptarget);
	do {
		const uint32_t nonce = (*pnonce);
		// GPU HASH
		decred_cpu_hash_nonce(thr_id, throughput, nonce);
		cudaDeviceSynchronize();
		if (h_resNonce[thr_id][0] != UINT32_MAX)
		{
			uint32_t vhashcpu[8];
			uint32_t Htarg = ptarget[6];

			if (opt_benchmark)
			{
				gpulog(LOG_WARNING, thr_id, "Found nonce: %0", h_resNonce[thr_id][0]);
			}

			be32enc(&endiandata[thr_id][DCR_NONCE_OFT32], h_resNonce[thr_id][0]);
			decred_hash(vhashcpu, endiandata[thr_id]);
			if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget))
			{
				rc = 1;
				*hashes_done = (*pnonce) - first_nonce + throughput;
				work_set_target_ratio(work, vhashcpu);
				work->nonces[0] = swab32(h_resNonce[thr_id][0]);
#if NBN > 1
				if (h_resNonce[thr_id][1] != UINT32_MAX) {
					be32enc(&endiandata[thr_id][DCR_NONCE_OFT32], h_resNonce[thr_id][1]);
					decred_hash(vhashcpu, endiandata[thr_id]);
					if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget)) {
						work->nonces[1] = swab32(h_resNonce[thr_id][1]);
						if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio) {
							work_set_target_ratio(work, vhashcpu);
							xchg(work->nonces[1], work->nonces[0]);
						}
						rc = 2;
					}
					h_resNonce[thr_id][1] = UINT32_MAX;
				}
#endif
				*pnonce = work->nonces[0];
				return rc;
			}
			else
			{
				if (vhashcpu[7] != 0)
				{
					applog_hash(ptarget);
					applog_compare_hash(vhashcpu, ptarget);
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[1]);
				}
			}
		}

		*pnonce += throughput;

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + (*pnonce));

	*hashes_done = (*pnonce) - first_nonce;
	return rc;
}

// cleanup
extern "C" void free_decred(int thr_id)
{
	if (!init[thr_id])
		return;

//	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

//	cudaDeviceSynchronize();
}

