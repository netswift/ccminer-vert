#ifndef __MINER_H__
#define __MINER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "cpuminer-config.h"

#include <stdbool.h>
#include <inttypes.h>
#include <sys/time.h>
#include <pthread.h>
#include <jansson.h>
#include <curl/curl.h>

#ifdef WIN32
#define snprintf(...) _snprintf(__VA_ARGS__)
#define strdup(x) _strdup(x)
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#define strcasecmp(x,y) _stricmp(x,y)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#undef HAVE_ALLOCA_H
#undef HAVE_SYSLOG_H
#endif

#ifdef STDC_HEADERS
# include <stdlib.h>
# include <stddef.h>
#else
# ifdef HAVE_STDLIB_H
#  include <stdlib.h>
# endif
#endif

#ifdef HAVE_ALLOCA_H
# include <alloca.h>
#elif !defined alloca
# ifdef __GNUC__
#  define alloca __builtin_alloca
# elif defined _AIX
#  define alloca __alloca
# elif defined _MSC_VER
#  include <malloc.h>
#  define alloca _alloca
# elif !defined HAVE_ALLOCA
#  ifdef  __cplusplus
extern "C"
#  endif
void *alloca (size_t);
# endif
#endif

#include "compat.h"
#include "log.h"

#ifdef __INTELLISENSE__
/* should be in stdint.h but... */
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef __int16 int8_t;
typedef unsigned __int16 uint8_t;

typedef unsigned __int32 time_t;
typedef char *  va_list;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
# define _ALIGN(x) __align__(x)
#elif _MSC_VER
# define _ALIGN(x) __declspec(align(x))
#else
# define _ALIGN(x) __attribute__ ((aligned(x)))
#endif

typedef unsigned char uchar;
#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifndef max
# define max(a, b)  ((a) > (b) ? (a) : (b))
#endif
#ifndef min
# define min(a, b)  ((a) < (b) ? (a) : (b))
#endif

#ifndef UINT32_MAX
/* for gcc 4.4 */
#define UINT32_MAX UINT_MAX
#endif

static inline bool is_windows(void) {
#ifdef WIN32
        return true;
#else
        return false;
#endif
}

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define WANT_BUILTIN_BSWAP
#else
#define bswap_32(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
                   | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#define bswap_64(x) (((uint64_t) bswap_32((uint32_t)((x) & 0xffffffffu)) << 32) \
                   | (uint64_t) bswap_32((uint32_t)((x) >> 32)))
#endif

static inline uint32_t swab32(uint32_t v)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap32(v);
#else
	return bswap_32(v);
#endif
}

static inline uint64_t swab64(uint64_t v)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap64(v);
#else
	return bswap_64(v);
#endif
}

static inline void swab256(void *dest_p, const void *src_p)
{
	uint32_t *dest = (uint32_t *) dest_p;
	const uint32_t *src = (const uint32_t *) src_p;

	dest[0] = swab32(src[7]);
	dest[1] = swab32(src[6]);
	dest[2] = swab32(src[5]);
	dest[3] = swab32(src[4]);
	dest[4] = swab32(src[3]);
	dest[5] = swab32(src[2]);
	dest[6] = swab32(src[1]);
	dest[7] = swab32(src[0]);
}

#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif

#if !HAVE_DECL_BE32DEC
static inline uint32_t be32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
	    ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}
#endif

#if !HAVE_DECL_LE32DEC
static inline uint32_t le32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
	    ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}
#endif

#if !HAVE_DECL_BE32ENC
static inline void be32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[3] = x & 0xff;
	p[2] = (x >> 8) & 0xff;
	p[1] = (x >> 16) & 0xff;
	p[0] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_LE32ENC
static inline void le32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
	p[2] = (x >> 16) & 0xff;
	p[3] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_BE16DEC
static inline uint16_t be16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[1]) + ((uint16_t)(p[0]) << 8));
}
#endif

#if !HAVE_DECL_BE16ENC
static inline void be16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[1] = x & 0xff;
	p[0] = (x >> 8) & 0xff;
}
#endif

#if !HAVE_DECL_LE16DEC
static inline uint16_t le16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[0]) + ((uint16_t)(p[1]) << 8));
}
#endif

#if !HAVE_DECL_LE16ENC
static inline void le16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
}
#endif

/* used for struct work */
void *aligned_calloc(int size);
void aligned_free(void *ptr);

#if JANSSON_MAJOR_VERSION >= 2
#define JSON_LOADS(str, err_ptr) json_loads((str), 0, (err_ptr))
#else
#define JSON_LOADS(str, err_ptr) json_loads((str), (err_ptr))
#endif

#define USER_AGENT PACKAGE_NAME "/" PACKAGE_VERSION

void sha256_init(uint32_t *state);
void sha256_transform(uint32_t *state, const uint32_t *block, int swap);
void sha256d(unsigned char *hash, const unsigned char *data, int len);

#if defined(__ARM_NEON__) || defined(__i386__) || defined(__x86_64__)
#define HAVE_SHA256_4WAY 0
int sha256_use_4way();
void sha256_init_4way(uint32_t *state);
void sha256_transform_4way(uint32_t *state, const uint32_t *block, int swap);
#endif

#if defined(__x86_64__) && defined(USE_AVX2)
#define HAVE_SHA256_8WAY 0
int sha256_use_8way();
void sha256_init_8way(uint32_t *state);
void sha256_transform_8way(uint32_t *state, const uint32_t *block, int swap);
#endif

extern int scanhash_sha256d(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_deep(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_doom(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_fugue256(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_groestlcoin(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_heavy(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done, uint32_t maxvote, int blocklen);

extern int scanhash_skeincoin(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_c11(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_keccak256(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_myriad(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_jackpot(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_quark(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_bastion(int thr_id, uint32_t *pdata, uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_blake256(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done, int8_t blakerounds);

extern int scanhash_fresh(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_lyra2(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_lyra2v2(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_nist5(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_pentablake(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_qubit(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_scrypt(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, unsigned char *scratchbuf, uint32_t max_nonce,
	unsigned long *hashes_done, struct timeval *tv_start, struct timeval *tv_end);

extern int scanhash_scrypt_jane(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, unsigned char *scratchbuf, uint32_t max_nonce,
	unsigned long *hashes_done, struct timeval *tv_start, struct timeval *tv_end);

extern int scanhash_s3(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_whc(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_whirlpoolx(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);


extern int scanhash_x11(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_x13(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_x14(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_x15(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_x17(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_bitcoin(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_neoscrypt(int stratum, int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_yescrypt(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	unsigned long *hashes_done);

extern int scanhash_bitcredit(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, const uint32_t *midstate, uint32_t max_nonce,
	unsigned long *hashes_done);

/* api related */
void *api_thread(void *userdata);
void api_set_throughput(int thr_id, uint32_t throughput);

struct cgpu_info {
	uint8_t gpu_id;
	uint8_t thr_id;
	int accepted;
	int rejected;
	int hw_errors;
	double khashes;
	uint8_t intensity_int;
	uint8_t has_monitoring;
	float gpu_temp;
	uint16_t gpu_fan;
	uint16_t gpu_fan_rpm;
	uint16_t gpu_arch;
	int gpu_clock;
	int gpu_memclock;
	size_t gpu_mem;
	uint32_t gpu_usage;
	double gpu_vddc;
	int16_t gpu_pstate;
	int16_t gpu_bus;
	uint16_t gpu_vid;
	uint16_t gpu_pid;

	int8_t nvml_id;
	int8_t nvapi_id;

	char gpu_sn[64];
	char gpu_desc[64];
	float intensity;
	uint32_t throughput;
};

struct thr_api {
	int id;
	pthread_t pth;
	struct thread_q	*q;
};

struct stats_data {
	uint32_t uid;
	uint32_t tm_stat;
	uint32_t hashcount;
	uint32_t height;
	double difficulty;
	double hashrate;
	uint8_t thr_id;
	uint8_t gpu_id;
	uint8_t hashfound;
	uint8_t ignored;
};

struct hashlog_data {
	uint32_t tm_sent;
	uint32_t height;
	uint32_t njobid;
	uint32_t nonce;
	uint32_t scanned_from;
	uint32_t scanned_to;
	uint32_t last_from;
	uint32_t tm_add;
	uint32_t tm_upd;
};

/* end of api */

struct thr_info {
	int		id;
	pthread_t	pth;
	struct thread_q	*q;
	struct cgpu_info gpu;
};

struct work_restart {
	volatile unsigned long	restart;
	char			padding[128 - sizeof(unsigned long)];
};

extern bool opt_eco_mode;
extern bool opt_benchmark;
extern bool opt_debug;
extern bool opt_quiet;
extern bool opt_protocol;
extern bool opt_tracegpu;
extern int opt_n_threads;
extern int opt_n_gputhreads;
extern bool opt_cpumining;
extern int num_cpus;
extern int active_gpus;
extern int opt_timeout;
extern bool want_longpoll;
extern bool have_longpoll;
extern bool want_stratum;
extern bool have_stratum;
extern char *opt_cert;
extern char *opt_proxy;
extern long opt_proxy_type;
extern bool use_syslog;
extern bool use_colors;
extern struct thr_info *thr_info;
extern int longpoll_thr_id;
extern int stratum_thr_id;
extern int api_thr_id;
extern struct work_restart *work_restart;
extern bool opt_trust_pool;
extern uint16_t opt_vote;

extern uint64_t global_hashrate;
extern double   global_diff;

extern bool scan_abort_flag;

#define MAX_GPUS 32
extern char* device_name[MAX_GPUS];
extern int device_map[MAX_GPUS];
extern long  device_sm[MAX_GPUS];
extern uint32_t gpus_intensity[MAX_GPUS];

extern void format_hashrate(double hashrate, char *output);
extern void applog(int prio, const char *fmt, ...);
extern json_t *json_rpc_call(CURL *curl, const char *url, const char *userpass,
	const char *rpc_req, bool, bool, int *);
extern void cbin2hex(char *out, const char *in, size_t len);
extern char *bin2hex(const unsigned char *in, size_t len);
extern bool hex2bin(unsigned char *p, const char *hexstr, size_t len);
extern int timeval_subtract(struct timeval *result, struct timeval *x,
	struct timeval *y);
extern bool fulltest(const uint32_t *hash, const uint32_t *target);
extern void diff_to_target(uint32_t *target, double diff);
extern void get_currentalgo(char* buf, int sz);
extern uint32_t device_intensity(int thr_id, const char *func, uint32_t defcount);

struct stratum_job {
	char *job_id;
	unsigned char prevhash[32];
	size_t coinbase_size;
	unsigned char *coinbase;
	unsigned char *xnonce2;
	int merkle_count;
	unsigned char **merkle;
	unsigned char version[4];
	unsigned char nbits[4];
	unsigned char ntime[4];
	bool clean;
	unsigned char nreward[2];
	uint32_t height;
	double diff;
};

struct stratum_ctx {
	char *url;

	CURL *curl;
	char *curl_url;
	char curl_err_str[CURL_ERROR_SIZE];
	curl_socket_t sock;
	size_t sockbuf_size;
	char *sockbuf;
	pthread_mutex_t sock_lock;

	double next_diff;

	char *session_id;
	size_t xnonce1_size;
	unsigned char *xnonce1;
	size_t xnonce2_size;
	struct stratum_job job;
	pthread_mutex_t work_lock;

	struct timeval tv_submit;
	uint32_t answer_msec;
	uint32_t disconnects;
	time_t tm_connected;

	int srvtime_diff;
};

struct work {
	uint32_t data[64]; //42 needed for bitcredit
	uint32_t target[8];
	uint32_t maxvote;
	uint32_t midstate[8];
	char job_id[128];
	size_t xnonce2_len;
	uchar xnonce2[32];

	union {
		uint32_t u32[2];
		uint64_t u64[1];
	} noncerange;

	double difficulty;
	uint32_t height;

	uint32_t scanned_from;
	uint32_t scanned_to;
};

bool stratum_socket_full(struct stratum_ctx *sctx, int timeout);
bool stratum_send_line(struct stratum_ctx *sctx, char *s);
char *stratum_recv_line(struct stratum_ctx *sctx);
bool stratum_connect(struct stratum_ctx *sctx, const char *url);
void stratum_disconnect(struct stratum_ctx *sctx);
bool stratum_subscribe(struct stratum_ctx *sctx);
bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass,bool extranonce);
bool stratum_handle_method(struct stratum_ctx *sctx, const char *s);

void hashlog_remember_submit(struct work* work, uint32_t nonce);
void hashlog_remember_scan_range(struct work* work);
uint32_t hashlog_already_submittted(char* jobid, uint32_t nounce);
uint32_t hashlog_get_last_sent(char* jobid);
uint64_t hashlog_get_scan_range(char* jobid);
int  hashlog_get_history(struct hashlog_data *data, int max_records);
void hashlog_purge_old(void);
void hashlog_purge_job(char* jobid);
void hashlog_purge_all(void);
void hashlog_dump_job(char* jobid);
void hashlog_getmeminfo(uint64_t *mem, uint32_t *records);

void stats_remember_speed(int thr_id, uint32_t hashcount, double hashrate, uint8_t found, uint32_t height);
double stats_get_speed(int thr_id, double def_speed);
int  stats_get_history(int thr_id, struct stats_data *data, int max_records);
void stats_purge_old(void);
void stats_purge_all(void);
void stats_getmeminfo(uint64_t *mem, uint32_t *records);

struct thread_q;

extern struct thread_q *tq_new(void);
extern void tq_free(struct thread_q *tq);
extern bool tq_push(struct thread_q *tq, void *data);
extern void *tq_pop(struct thread_q *tq, const struct timespec *abstime);
extern void tq_freeze(struct thread_q *tq);
extern void tq_thaw(struct thread_q *tq);

void proper_exit(int reason);

size_t time2str(char* buf, time_t timer);
char* atime2str(time_t timer);

void print_hash_tests(void);
void blake256hash(void *output, const void *input, int8_t rounds);
void deephash(void *state, const void *input);
void doomhash(void *state, const void *input);
void fresh_hash(void *state, const void *input);
void fugue256_hash(unsigned char* output, const unsigned char* input, int len);
void heavycoin_hash(unsigned char* output, const unsigned char* input, int len);
void keccak256_hash(void *state, const void *input);
uint32_t jackpothash(void *state, const void *input);
void groestlhash(void *state, const void *input);
void lyra2_hash(void *state, const void *input);
void myriadhash(void *state, const void *input);
void nist5hash(void *state, const void *input);
void pentablakehash(void *output, const void *input);
void quarkhash(void *state, const void *input);
void qubithash(void *state, const void *input);
void scrypthash(void* output, const void* input);
void skeincoinhash(void *output, const void *input);
void s3hash(void *output, const void *input);
void wcoinhash(void *state, const void *input);
void x11hash(void *output, const void *input);
void x13hash(void *output, const void *input);
void x14hash(void *output, const void *input);
void x15hash(void *output, const void *input);
void x17hash(void *output, const void *input);

#ifdef __cplusplus
}
#endif

#endif /* __MINER_H__ */
