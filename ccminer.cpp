﻿/*
 * Copyright 2010 Jeff Garzik
 * Copyright 2012-2014 pooler
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "cpuminer-config.h"
#include "cuda_runtime_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <signal.h>
#include <curl/curl.h>
#include <jansson.h>
#include <openssl/sha.h>
#ifdef WIN32
#include <windows.h>
#include <stdint.h>
#else
#include <errno.h>
#include <sys/resource.h>
#if HAVE_SYS_SYSCTL_H
#include <sys/types.h>
#if HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#include <sys/sysctl.h>
#endif
#endif

//#include "nvml.h"
#include "cuda_runtime.h"
//#include "gpu_utils.h"
#include "miner.h"

#ifdef WIN32
#include <Mmsystem.h>
#pragma comment(lib, "winmm.lib")
#include "compat/winansi.h"
BOOL WINAPI ConsoleHandler(DWORD);
#endif

#define PROGRAM_NAME		"ccminer"
#define LP_SCANTIME		30
#define HEAVYCOIN_BLKHDR_SZ		84
#define MNR_BLKHDR_SZ 80

// from cuda.cpp
int cuda_num_devices();
void cuda_devicenames();
void cuda_devicenames();
void cuda_devicereset();
void cuda_print_devices();
int cuda_finddevice(char *name);

#include "nvml.h"
#ifdef USE_WRAPNVML
nvml_handle *hnvml = NULL;
#endif

enum workio_commands {
	WC_GET_WORK,
	WC_SUBMIT_WORK,
	WC_ABORT,
};

struct workio_cmd {
	enum workio_commands	cmd;
	struct thr_info		*thr;
	union {
		struct work	*work;
	} u;
};

enum sha_algos {
	ALGO_BASTION,
	ALGO_BITC,
	ALGO_BITCOIN,
	ALGO_BLAKE,
	ALGO_BLAKECOIN,
	ALGO_C11,
	ALGO_DEEP,
	ALGO_DMD_GR,
	ALGO_DOOM,
	ALGO_FRESH,
	ALGO_FUGUE256,		/* Fugue256 */
	ALGO_GROESTL,
	ALGO_HEAVY,		/* Heavycoin hash */
	ALGO_KECCAK,
	ALGO_JACKPOT,
	ALGO_LUFFA_DOOM,
	ALGO_LYRA2,
	ALGO_LYRA2v2,
	ALGO_MJOLLNIR,		/* Hefty hash */
	ALGO_MYR_GR,
	ALGO_NIST5,
	ALGO_PENTABLAKE,
	ALGO_QUARK,
	ALGO_QUBIT,
	ALGO_SCRYPT,
	ALGO_SCRYPT_JANE, 
	ALGO_SKEIN,
	ALGO_S3,
	ALGO_SPREADX11,
	ALGO_WHC,
	ALGO_WHCX,
	ALGO_X11,
	ALGO_X13,
	ALGO_X14,
	ALGO_X15,
	ALGO_X17,
	ALGO_VANILLA,
	ALGO_NEO,
	ALGO_COUNT

};

static const char *algo_names[] = {
	"bastion",
	"credit",
	"bitcoin",
	"blake",
	"blakecoin",
	"c11", 
	"deep",
	"dmd-gr",
	"doom", /* is luffa */
	"fresh",
	"fugue256",
	"groestl",
	"heavy",
	"keccak",
	"jackpot",
	"luffa",
	"lyra2",
	"lyra2v2",
	"mjollnir",
	"myr-gr",
	"nist5",
	"penta",
	"quark",
	"qubit",
	"scrypt",
	"scrypt-jane",
	"skein",
	"s3",
	"spread",
	"whirl",
	"whirlpoolx",
	"x11",
	"x13",
	"x14",
	"x15",
	"x17",
	"vanilla",
	"neoscrypt",
	""
};

bool opt_broken_neo_wallet = false; // HACK: Drop this flag once neoscrypt wallets are fixed to send 80byte data
bool opt_eco_mode = false;
bool opt_debug = false;
bool opt_protocol = false;
bool opt_benchmark = false;
bool want_longpoll = true;
bool have_longpoll = false;
bool want_stratum = true;
bool have_stratum = false;
bool allow_gbt = false;
bool check_dups = false;
static bool submit_old = false;
bool use_syslog = false;
bool use_colors = true;
static bool opt_background = false;
bool opt_quiet = false;
static int opt_retries = -1;
static int opt_fail_pause = 5;
static int opt_time_limit = 0;
int opt_timeout = 270;
static int opt_scantime = 5;
static json_t *opt_config;
static const bool opt_time = true;
static enum sha_algos opt_algo = ALGO_X11;
int opt_n_threads = 0;
int opt_n_gputhreads = 1;
int opt_affinity = -1;
int opt_priority = 0;
static double opt_difficulty = 1; // CH
static bool opt_extranonce = true;
bool opt_cpumining = false;

bool opt_trust_pool = false;
uint16_t opt_vote = 9999;
int num_cpus = 0;
int active_gpus = 0;
char * device_name[MAX_GPUS];
int device_map[MAX_GPUS] = { 0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 };
long  device_sm[MAX_GPUS] = { 0 };
int device_memspeed[MAX_GPUS] = { 0 };
int device_gpuspeed[MAX_GPUS] = { 0 };
uint32_t gpus_intensity[MAX_GPUS] = { 0 };
int device_interactive[MAX_GPUS] = { 0 };
int device_batchsize[MAX_GPUS] = { 0 };
int device_backoff[MAX_GPUS] = { 0 };
int device_lookup_gap[MAX_GPUS] = { 0 };
int device_texturecache[MAX_GPUS] = { 0 };
int device_singlememory[MAX_GPUS] = { 0 };


char *device_config[MAX_GPUS] = { 0 };
int opt_nfactor = 0;
int parallel = 2;
bool autotune = true;
bool opt_autotune = true;

bool abort_flag = false;
bool scan_abort_flag = false;
bool network_fail_flag = false;
char *jane_params = NULL;

char *rpc_user = NULL;
static char *rpc_url = NULL;
static char *rpc_userpass = NULL;
static char *rpc_pass = NULL;
static char *short_url = NULL;
char *opt_cert = NULL;
char *opt_proxy = NULL;
long opt_proxy_type = -1;
struct thr_info *thr_info = NULL;
static int work_thr_id = -1;
struct thr_api *thr_api = NULL;
int longpoll_thr_id = -1;
int stratum_thr_id = -1;
int api_thr_id = -1;
bool stratum_need_reset = false;
struct work_restart *work_restart = NULL;
struct stratum_ctx stratum = { 0 };

static pthread_mutex_t stats_lock = PTHREAD_MUTEX_INITIALIZER;
uint32_t accepted_count = 0L;
uint32_t rejected_count = 0L;
static double thr_hashrates[MAX_GPUS] = { 0 };
uint64_t global_hashrate = 0;
double   global_diff = 0.0;
uint32_t opt_statsavg = 30;
static char* opt_syslog_pfx = NULL;
char *opt_api_allow = NULL;
int opt_api_listen = 0; /* 0 to disable */

#ifdef HAVE_GETOPT_LONG
#include <getopt.h>
#else
struct option {
	const char *name;
	int has_arg;
	int *flag;
	int val;
};
#endif

static char const usage[] = "\
Usage: " PROGRAM_NAME " [OPTIONS]\n\
Options:\n\
  -a, --algo=ALGO       specify the hash algorithm to use\n\
			bastion		bastioncoin\n\
			bitcoin     Bitcoin\n\
			blake       Blake 256 (SFR/NEOS)\n\
			blakecoin   Fast Blake 256 (8 rounds)\n\
			c11/flax    X11 variant\n\
            credit      Credit\n\
			deep        Deepcoin\n\
			dmd-gr      Diamond-Groestl\n\
			fresh       Freshcoin (shavite 80)\n\
			fugue256    Fuguecoin\n\
			groestl     Groestlcoin\n\
			heavy       Heavycoin\n\
			jackpot     Jackpot\n\
			keccak      Keccak-256 (Maxcoin)\n\
			luffa       Doomcoin\n\
			lyra2       Lyrabar\n\
			lyra2v2     VertCoin\n\
			mjollnir    Mjollnircoin\n\
			myr-gr      Myriad-Groestl\n\
            neoscrypt   neoscrypt (FeatherCoin)\n\
			nist5       NIST5 (TalkCoin)\n\
			penta       Pentablake hash (5x Blake 512)\n\
			quark       Quark\n\
			qubit       qubit\n\
			scrypt      Scrypt\n\
			scrypt-jane Scrypt-jane Chacha\n\
			skein       Skein\n\
			s3          S3 (1Coin)\n\
			spread      Spread\n\
			x11         X11 (DarkCoin)\n\
			x13         X13 (MaruCoin)\n\
			x14         X14\n\
			x15         X15\n\
			x17         X17 (peoplecurrency)\n\
			Vanilla (Blake256 8-rounds - double sha256)\n\
			yescrypt    yescrypt\n\
			whirl       Whirlcoin (old whirlpool)\n\
			whirlpoolx  Vanillacoin \n\
  -d, --devices         Comma separated list of CUDA devices to use. \n\
                        Device IDs start counting from 0! Alternatively takes\n\
                        string names of your cards like gtx780ti or gt640#2\n\
                        (matching 2nd gt640 in the PC)\n\
  -i  --intensity=N     GPU intensity 8-31 (default: auto) \n\
                        Decimals are allowed for fine tuning \n\
  -f, --diff            Divide difficulty by this factor (std is 1) \n\
  -v, --vote=VOTE       block reward vote (for HeavyCoin)\n\
  -m, --trust-pool      trust the max block reward vote (maxvote) sent by the pool\n\
  -o, --url=URL         URL of mining server\n\
  -O, --userpass=U:P    username:password pair for mining server\n\
  -u, --user=USERNAME   username for mining server\n\
  -p, --pass=PASSWORD   password for mining server\n\
      --cert=FILE       certificate for mining server using SSL\n\
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy\n\
  -t, --threads=N       number of miner threads (default: number of nVidia GPUs)\n\
  -g, --gputhreads=N    number of threads per gpu (default: 1)\n\
  -r, --retries=N       number of times to retry if a network call fails\n\
                          (default: retry indefinitely)\n\
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 30)\n\
      --time-limit      maximum time [s] to mine before exiting the program.\n\
  -T, --timeout=N       network timeout, in seconds (default: 270)\n\
  -s, --scantime=N      upper bound on time spent scanning current work when\n\
                          long polling is unavailable, in seconds (default: 5)\n\
  -n, --ndevs           list cuda devices\n\
  -N, --statsavg        number of samples used to display hashrate (default: 30)\n\
      --no-gbt          disable getblocktemplate support (height check in solo)\n\
      --no-longpoll     disable X-Long-Polling support\n\
      --no-stratum      disable X-Stratum support\n\
  -q, --quiet           disable per-thread hashmeter output\n\
      --no-color        disable colored output\n\
  -D, --debug           enable debug output\n\
  -P, --protocol-dump   verbose dump of protocol-level activities\n\
      --cpu-affinity    set process affinity to cpu core(s), mask 0x3 for cores 0 and 1\n\
      --cpu-priority    set process priority (default: 0 idle, 2 normal to 5 highest)\n\
  -b, --api-bind        IP/Port for the miner API (default: 127.0.0.1:4068)\n\
  -S, --syslog          use system log for output messages\n\
  --syslog - prefix = ... allow to change syslog tool name\n\
   -B, --background      run the miner in the background\n\
		--benchmark       run in offline benchmark mode\n\
      --cputest         debug hashes from cpu algorithms\n\
  -c, --config=FILE     load a JSON-format configuration file\n\
  -C, --cpu-mining		Enable the cpu to aid the gpu. (warning: uses more power)\n\
  -V, --version         display version information and exit\n\
  -h, --help            display this help text and exit\n\
  -X,  --XIntensity     intensity GPU intensity(default: auto) \n\
      --broken-neo-wallet	Use 84byte data for broken neoscrypt wallets.\n\
      --eco				Use Eco mode.\n\
";

char const short_options[] = "SX:a:c:i:Dhp:Px:qr:R:s:t:T:o:u:O:Vd:f:mv:N:n:b:g:l:L:e:M:C";

struct option const options[] = {
	{ "algo", 1, NULL, 'a' },
	{ "api-bind", 1, NULL, 'b' },
	{ "benchmark", 0, NULL, 1005 },
	{ "cert", 1, NULL, 1001 },
	{ "config", 1, NULL, 'c' },
	{ "cputest", 0, NULL, 1006 },
	{ "cpu-affinity", 1, NULL, 1020 },
	{ "cpu-priority", 1, NULL, 1021 },
	{ "debug", 0, NULL, 'D' },
	{ "help", 0, NULL, 'h' },
	{ "intensity", 1, NULL, 'i' },
	{ "ndevs", 0, NULL, 'n' },
	{ "no-color", 0, NULL, 1002 },
	{ "no-gbt", 0, NULL, 1011 },
	{ "no-longpoll", 0, NULL, 1003 },
	{ "no-stratum", 0, NULL, 1007 },
	{ "no-autotune", 0, NULL, 1004 },  // scrypt
	{ "launch-config", 0, NULL, 'l' }, // scrypt
	{ "lookup-gap", 0, NULL, 'L' },    // scrypt
	{ "pass", 1, NULL, 'p' },
	{ "protocol-dump", 0, NULL, 'P' },
	{ "proxy", 1, NULL, 'x' },
	{ "quiet", 0, NULL, 'q' },
	{ "retries", 1, NULL, 'r' },
	{ "retry-pause", 1, NULL, 'R' },
	{ "syslog", 0, NULL, 'S' },
	{ "syslog-prefix", 1, NULL, 1008 },
	{ "scantime", 1, NULL, 's' },
	{ "statsavg", 1, NULL, 'N' },
	{ "time-limit", 1, NULL, 1008 },
	{ "threads", 1, NULL, 't' },
	{ "gputhreads", 1, NULL, 'g' },
	{ "gpu-engine", 1, NULL, 1070 },
	{ "gpu-memclock", 1, NULL, 1071 },
	{ "Disable extranounce support", 1, NULL, 'e' },
	{ "vote", 1, NULL, 'v' },
	{ "trust-pool", 0, NULL, 'm' },
	{ "timeout", 1, NULL, 'T' },
	{ "url", 1, NULL, 'o' },
	{ "user", 1, NULL, 'u' },
	{ "userpass", 1, NULL, 'O' },
	{ "version", 0, NULL, 'V' },
	{ "devices", 1, NULL, 'd' },
	{ "diff", 1, NULL, 'f' },
	{ "X", 1, NULL, 'X'},
	{ "cpu-mining", 0, NULL, 'C'},
	{ "broken-neo-wallet", 0, NULL, 1030 },
	{ "eco", 0, NULL, 1080 },
	{ 0, 0, 0, 0 }
};

static char const scrypt_usage[] = "\n\
									Scrypt specific options:\n\
									  -l, --launch-config   gives the launch configuration for each kernel\n\
									                        in a comma separated list, one per device.\n\
									  -L, --lookup-gap      Divides the per-hash memory requirement by this factor\n\
									                        by storing only every N'th value in the scratchpad.\n\
									                        Default is 1.\n\
									      --no-autotune     disable auto-tuning of kernel launch parameters\n\
									";

static struct work _ALIGN(64) g_work;
static time_t g_work_time;
static pthread_mutex_t g_work_lock = PTHREAD_MUTEX_INITIALIZER;


#ifdef __linux /* Linux specific policy and affinity management */
#include <sched.h>
static inline void drop_policy(void) {
	struct sched_param param;
	param.sched_priority = 0;
#ifdef SCHED_IDLE
	if (unlikely(sched_setscheduler(0, SCHED_IDLE, &param) == -1))
#endif
#ifdef SCHED_BATCH
		sched_setscheduler(0, SCHED_BATCH, &param);
#endif
}
static void affine_to_cpu_mask(int id, uint8_t mask) {
	cpu_set_t set;
	CPU_ZERO(&set);
	for (uint8_t i = 0; i < num_cpus; i++) {
		// cpu mask
		if (mask & (1<<i)) { CPU_SET(i, &set); }
	}
	if (id == -1) {
		// process affinity
		sched_setaffinity(0, sizeof(&set), &set);
	} else {
		// thread only
		pthread_setaffinity_np(thr_info[id].pth, sizeof(&set), &set);
	}
}
#elif defined(__FreeBSD__) /* FreeBSD specific policy and affinity management */
#include <sys/cpuset.h>
static inline void drop_policy(void) { }
static void affine_to_cpu_mask(int id, uint8_t mask) {
	cpuset_t set;
	CPU_ZERO(&set);
	for (uint8_t i = 0; i < num_cpus; i++) {
		if (mask & (1<<i)) CPU_SET(i, &set);
	}
	cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(cpuset_t), &set);
}
#else /* Windows */
static inline void drop_policy(void) { }
static void affine_to_cpu_mask(int id, uint8_t mask) {
	if (id == -1)
		SetProcessAffinityMask(GetCurrentProcess(), mask);
	else
		SetThreadAffinityMask(GetCurrentThread(), mask);
}
#endif

static bool get_blocktemplate(CURL *curl, struct work *work);

void get_currentalgo(char* buf, int sz)
{
	snprintf(buf, sz, "%s", algo_names[opt_algo]);
}

/**
 * Exit app
 */

#define CCEXIT_SIG -1
void proper_exit(int reason)
{
	struct thr_info* thr = NULL;

	abort_flag = true;

        if (thr_info && work_thr_id != -1)
          thr = &thr_info[work_thr_id];
        if (thr && thr->q)
          tq_freeze(thr->q);

	if (reason != CCEXIT_SIG) {
#ifdef USE_WRAPNVML
		if (hnvml)
			nvml_destroy(hnvml);
#endif

		free(opt_syslog_pfx);
		opt_syslog_pfx = NULL;
		free(opt_api_allow);
		opt_api_allow = NULL;
		hashlog_purge_all();
		stats_purge_all();
		cuda_devicereset();
	}
}

static bool jobj_binary(const json_t *obj, const char *key,
			void *buf, size_t buflen)
{
	const char *hexstr;
	json_t *tmp;

	tmp = json_object_get(obj, key);
	if (unlikely(!tmp)) {
		applog(LOG_ERR, "JSON key '%s' not found", key);
		return false;
	}
	hexstr = json_string_value(tmp);
	if (unlikely(!hexstr)) {
		applog(LOG_ERR, "JSON key '%s' is not a string", key);
		return false;
	}
	if (!hex2bin((uchar*)buf, hexstr, buflen))
		return false;

	return true;
}


static bool work_decode(const json_t *val, struct work *work)
{
	int data_size, midstate_size;
	switch (opt_algo) {
	case ALGO_NEO:
		data_size = opt_broken_neo_wallet ? 84 : 80;
		break;
	case ALGO_BITC:
		data_size = 168;
		midstate_size = sizeof(work->midstate);
		break;
	default:
		data_size = 128; // original sizeof(work->data); however data is now 64*4bit
	}
	int adata_sz = data_size >> 2;
	int target_size = sizeof(work->target);
	int atarget_sz = target_size >> 2;
	int i;
	//	printf("data size %d",data_size);
	if (unlikely(!jobj_binary(val, "data", work->data, data_size))) {
		applog(LOG_ERR, "JSON inval data");
		return false;
	}
	if (unlikely(!jobj_binary(val, "target", work->target, target_size))) {
		applog(LOG_ERR, "JSON inval target");
		return false;
	}

	if (opt_algo == ALGO_BITC) {
		if (unlikely(!jobj_binary(val, "midstate", work->midstate, midstate_size))) {
			applog(LOG_ERR, "JSON inval midstate");
			return false;
		}

		for (i = 0; i < midstate_size>>2; i++)
			work->midstate[i] = le32dec(work->midstate + i);
	}


	if (opt_algo == ALGO_HEAVY) {
		if (unlikely(!jobj_binary(val, "maxvote", &work->maxvote, sizeof(work->maxvote)))) {
			work->maxvote = 2048;
		}
	}
	else work->maxvote = 0;
	//	printf("the data: \n");
	for (i = 0; i < adata_sz; i++)
		work->data[i] = le32dec(work->data + i);

	//		printf("i %d %08x \n",i, work->data[i]);}
	//	printf("\n");
	for (i = 0; i < atarget_sz; i++)
		work->target[i] = le32dec(work->target + i);

	json_t *jr = json_object_get(val, "noncerange");
	if (jr) {
		const char * hexstr = json_string_value(jr);
		if (likely(hexstr)) {
			// never seen yet...
			hex2bin((uchar*)work->noncerange.u64, hexstr, 8);
			applog(LOG_DEBUG, "received noncerange: %08x-%08x",
				work->noncerange.u32[0], work->noncerange.u32[1]);
		}
	}

	/* use work ntime as job id (solo-mining) */
	cbin2hex(work->job_id, (const char*)&work->data[17], 4);

	return true;
}

/**
 * Calculate the work difficulty as double
 * Not sure it works with pools
 */
static void calc_diff(struct work *work, int known)
{
	// sample for diff 32.53 : 00000007de5f0000
	const uint64_t diffone = 0xFFFF000000000000ull;
	uint64_t *data64, d64;
	char rtarget[32];

	swab256(rtarget, work->target);
	data64 = (uint64_t *)(rtarget + 3); /* todo: index (3) can be tuned here */

	if (opt_algo == ALGO_HEAVY) {
		data64 = (uint64_t *)(rtarget + 2);
	}

	d64 = swab64(*data64);
	if (unlikely(!d64))
		d64 = 1;
	work->difficulty = (double)diffone / d64;
	if (opt_difficulty > 0.) {
		work->difficulty /= opt_difficulty;
	}
}

static int share_result(int result, const char *reason)
{
	char s[345];
	double hashrate = 0.;
	const char *sres;

	pthread_mutex_lock(&stats_lock);

	for (int i = 0; i < opt_n_threads; i++) {
		hashrate += stats_get_speed(i, thr_hashrates[i]);
	}

	result ? accepted_count++ : rejected_count++;
	pthread_mutex_unlock(&stats_lock);

	global_hashrate = llround(hashrate);

	if (use_colors)
		sres = (result ? CL_GRN "yes!" : CL_RED "nooooo");
	else
		sres = (result ? "(yes!!!)" : "(nooooo)");

	switch (opt_algo) {
	case ALGO_BASTION:
		sprintf(s, hashrate >= 1e6 ? "%.0f" : "%.2f", hashrate);
		applog(LOG_NOTICE, "accepted: %lu/%lu (%.2f%%), %s H/s %s",
			accepted_count, accepted_count + rejected_count,
			100. * accepted_count / (accepted_count + rejected_count), s, sres);
		break;
	default:
		sprintf(s, hashrate >= 1e6 ? "%.0f" : "%.2f", hashrate / 1000.0);
		applog(LOG_NOTICE, "accepted: %lu/%lu (%.2f%%), %s kH/s %s",
			accepted_count, accepted_count + rejected_count,
			100. * accepted_count / (accepted_count + rejected_count), s, sres);
		break;
	}

	if (reason) {
		applog(LOG_WARNING, "reject reason: %s", reason);
		if (strncmp(reason, "Duplicate share", 15) == 0 && !check_dups) {
			applog(LOG_WARNING, "enabling duplicates check feature");
			check_dups = true;
		}
		return 0;

	}
	return 1;
}

static bool submit_upstream_work(CURL *curl, struct work *work)
{
	json_t *val, *res, *reason;
	bool stale_work = false;
	char s[384];

	/* discard if a newer bloc was received */
	/*
	stale_work = work->height && work->height < g_work.height;
	if (have_stratum && !stale_work) {
	pthread_mutex_lock(&g_work_lock);
	if (strlen(work->job_id + 8))
	stale_work = strcmp(work->job_id + 8, g_work.job_id + 8);
	pthread_mutex_unlock(&g_work_lock);
	}
	*/
	if (!have_stratum && !stale_work && allow_gbt) {
		struct work wheight = { 0 };
		if (get_blocktemplate(curl, &wheight)) {
			if (work->height && work->height < wheight.height) {
				if (opt_debug)
					applog(LOG_WARNING, "bloc %u was already solved", work->height, wheight.height);
				return true;
			}
		}
	}

	if (stale_work) {
		if (opt_debug)
			applog(LOG_WARNING, "stale work detected, discarding");
		return true;
	}
	calc_diff(work, 0);

	if (have_stratum) 
	{
		uint32_t sent = 0;
		uint32_t ntime, nonce;
		uint16_t nvote;
		char *ntimestr, *noncestr, *xnonce2str, *nvotestr;
		le32enc(&ntime, work->data[17]);
		le32enc(&nonce, work->data[19]);
		noncestr = bin2hex((const uchar*)(&nonce), 4);

		if (check_dups)
			sent = hashlog_already_submittted(work->job_id, nonce);
		if (sent > 0) {
			sent = (uint32_t)time(NULL) - sent;
			if (!opt_quiet) {
				applog(LOG_WARNING, "nonce %s was already sent %u seconds ago", noncestr, sent);
				hashlog_dump_job(work->job_id);
			}
			free(noncestr);
			// prevent useless computing on some pools
			stratum_need_reset = true;
			for (int i = 0; i < opt_n_threads; i++)
				work_restart[i].restart = 1;

			return true;
		}

		ntimestr = bin2hex((const uchar*)(&ntime), 4);
		xnonce2str = bin2hex(work->xnonce2, work->xnonce2_len);

		if (opt_algo == ALGO_HEAVY) {
			be16enc(&nvote, *((uint16_t*)&work->data[20]));
			nvotestr = bin2hex((const uchar*)(&nvote), 2);
			sprintf(s,
				"{\"method\": \"mining.submit\", \"params\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\"], \"id\":4}",
				rpc_user, work->job_id + 8, xnonce2str, ntimestr, noncestr, nvotestr);
			free(nvotestr);
		}
		else {
			sprintf(s,
				"{\"method\": \"mining.submit\", \"params\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%s\"], \"id\":4}",
				rpc_user, work->job_id + 8, xnonce2str, ntimestr, noncestr);
		}
		free(xnonce2str);
		free(ntimestr);
		free(noncestr);

		gettimeofday(&stratum.tv_submit, NULL);
		if (unlikely(!stratum_send_line(&stratum, s))) {
			applog(LOG_ERR, "submit_upstream_work stratum_send_line failed");
			sleep(10);
			return false;
		}

		if (check_dups)
			hashlog_remember_submit(work, nonce);

	}
	else {

		/* build hex string */
		char *str = NULL;
		int data_size;

		switch (opt_algo)
		{
		case ALGO_NEO:
			data_size = opt_broken_neo_wallet ? 84 : 80;
			break;
		case ALGO_BITC:
			data_size = 168;
			break;
		default:
			data_size = 128;
			break;
		}

		if (opt_algo != ALGO_HEAVY && opt_algo != ALGO_MJOLLNIR) {
			for (int i = 0; i < (data_size >> 2); i++)
				le32enc(work->data + i, work->data[i]);
		}
		str = bin2hex((uchar*)work->data, data_size);
		if (unlikely(!str)) {
			applog(LOG_ERR, "submit_upstream_work OOM");
			return false;
		}

		/* build JSON-RPC request */
		sprintf(s,
			"{\"method\": \"getwork\", \"params\": [\"%s\"], \"id\":4}\r\n",
			str);

		/* issue JSON-RPC request */
		val = json_rpc_call(curl, rpc_url, rpc_userpass, s, false, false, NULL);
		if (unlikely(!val)) {
			applog(LOG_ERR, "submit_upstream_work json_rpc_call failed");
			return false;
		}

		res = json_object_get(val, "result");
		reason = json_object_get(val, "reject-reason");
		if (!share_result(json_is_true(res), reason ? json_string_value(reason) : NULL)) {
			if (check_dups)
				hashlog_purge_job(work->job_id);
		}

		json_decref(val);

		free(str);
	}

	return true;
}

/* simplified method to only get some extra infos in solo mode */
static bool gbt_work_decode(const json_t *val, struct work *work)
{
	json_t *err = json_object_get(val, "error");
	if (err && !json_is_null(err)) {
		allow_gbt = false;
		applog(LOG_INFO, "GBT not supported, bloc height unavailable");
		return false;
	}

	if (!work->height) {
		// complete missing data from getwork
		json_t *key = json_object_get(val, "height");
		if (key && json_is_integer(key)) {
			work->height = (uint32_t) json_integer_value(key);
			if (!opt_quiet && work->height > g_work.height) {
				applog(LOG_BLUE, "%s %s block %d", short_url,
					algo_names[opt_algo], work->height);
				g_work.height = work->height;
			}
		}
	}

	return true;
}

#define GBT_CAPABILITIES "[\"coinbasetxn\", \"coinbasevalue\", \"longpoll\", \"workid\"]"
static const char *gbt_req =
	"{\"method\": \"getblocktemplate\", \"params\": ["
	//	"{\"capabilities\": " GBT_CAPABILITIES "}"
	"], \"id\":0}\r\n";

static bool get_blocktemplate(CURL *curl, struct work *work)
{
	if (!allow_gbt)
		return false;

	json_t *val = json_rpc_call(curl, rpc_url, rpc_userpass, gbt_req,
			    want_longpoll, false, NULL);

	if (!val)
		return false;

	bool rc = gbt_work_decode(json_object_get(val, "result"), work);

	json_decref(val);

	return rc;
}

static const char *rpc_req =
	"{\"method\": \"getwork\", \"params\": [], \"id\":0}\r\n";

static bool get_upstream_work(CURL *curl, struct work *work)
{
	json_t *val;
	bool rc;
	struct timeval tv_start, tv_end, diff;

	gettimeofday(&tv_start, NULL);
	val = json_rpc_call(curl, rpc_url, rpc_userpass, rpc_req,
			    want_longpoll, false, NULL);
	gettimeofday(&tv_end, NULL);

	if (have_stratum) {
		if (val)
			json_decref(val);
		return true;
	}

	if (!val)
		return false;

	rc = work_decode(json_object_get(val, "result"), work);

	if (opt_protocol && rc) {
		timeval_subtract(&diff, &tv_end, &tv_start);
		/* show time because curl can be slower against versions/config */
		applog(LOG_DEBUG, "got new work in %.2f ms",
		       (1000.0 * diff.tv_sec) + (0.001 * diff.tv_usec));
	}

	json_decref(val);

	get_blocktemplate(curl, work);

	return rc;
}

static void workio_cmd_free(struct workio_cmd *wc)
{
	if (!wc)
		return;

	switch (wc->cmd) {
	case WC_SUBMIT_WORK:
		aligned_free(wc->u.work);
		break;
	default: /* do nothing */
		break;
	}

	memset(wc, 0, sizeof(*wc));	/* poison */
	free(wc);
}

static void workio_abort()
{
	struct workio_cmd *wc;

	/* fill out work request message */
	wc = (struct workio_cmd *)calloc(1, sizeof(*wc));
	if (!wc)
		return;

	wc->cmd = WC_ABORT;

	/* send work request to workio thread */
	if (!tq_push(thr_info[work_thr_id].q, wc)) {
		workio_cmd_free(wc);
	}
}

static bool workio_get_work(struct workio_cmd *wc, CURL *curl)
{
	struct work *ret_work;
	int failures = 0;

	ret_work = (struct work*)aligned_calloc(sizeof(*ret_work));
	if (!ret_work)
		return false;

	/* obtain new work from bitcoin via JSON-RPC */
	while (!get_upstream_work(curl, ret_work)) {
		if (unlikely((opt_retries >= 0) && (++failures > opt_retries))) {
			applog(LOG_ERR, "json_rpc_call failed, terminating workio thread");
			aligned_free(ret_work);
			return false;
		}

		/* pause, then restart work-request loop */
		applog(LOG_ERR, "json_rpc_call failed, retry after %d seconds",
			opt_fail_pause);
		sleep(opt_fail_pause);
	}

	/* send work to requesting thread */
	if (!tq_push(wc->thr->q, ret_work))
		aligned_free(ret_work);

	return true;
}

static bool workio_submit_work(struct workio_cmd *wc, CURL *curl)
{
	int failures = 0;

	/* submit solution to bitcoin via JSON-RPC */
	while (!submit_upstream_work(curl, wc->u.work)) {
		if (unlikely((opt_retries >= 0) && (++failures > opt_retries))) {
			applog(LOG_ERR, "...terminating workio thread");
			return false;
		}

		/* pause, then restart work-request loop */
		if (!opt_benchmark)
			applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);

		sleep(opt_fail_pause);
	}

	return true;
}

static void *workio_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info*)userdata;
	CURL *curl;
	bool ok = true;

	curl = curl_easy_init();
	if (unlikely(!curl)) {
		applog(LOG_ERR, "CURL initialization failed");
		return NULL;
	}

	while (ok && !abort_flag) {
		struct workio_cmd *wc;

		/* wait for workio_cmd sent to us, on our queue */
		wc = (struct workio_cmd *)tq_pop(mythr->q, NULL);
		if (!wc) {
			ok = false;
			break;
		}

		/* process workio_cmd */
		switch (wc->cmd) {
		case WC_GET_WORK:
			ok = workio_get_work(wc, curl);
			break;
		case WC_SUBMIT_WORK:
			ok = workio_submit_work(wc, curl);
			break;
		case WC_ABORT:
		default:		/* should never happen */
			ok = false;
			break;
		}

		workio_cmd_free(wc);
	}

	tq_freeze(mythr->q);
	curl_easy_cleanup(curl);

	return NULL;
}

static bool get_work(struct thr_info *thr, struct work *work)
{
	struct workio_cmd *wc;
	struct work *work_heap;

	if (opt_benchmark) {
		memset(work->data, 0x55, 76);
		//work->data[17] = swab32((uint32_t)time(NULL));
		memset(work->data + 19, 0x00, 52);
		work->data[20] = 0x80000000;
		work->data[31] = 0x00000280;
		memset(work->target, 0x00, sizeof(work->target));
		return true;
	}

	/* fill out work request message */
	wc = (struct workio_cmd *)calloc(1, sizeof(*wc));
	if (!wc)
		return false;

	wc->cmd = WC_GET_WORK;
	wc->thr = thr;

	/* send work request to workio thread */
	if (!tq_push(thr_info[work_thr_id].q, wc)) {
		workio_cmd_free(wc);
		return false;
	}

	/* wait for response, a unit of work */
	work_heap = (struct work *)tq_pop(thr->q, NULL);
	if (!work_heap)
		return false;

	/* copy returned work into storage provided by caller */
	memcpy(work, work_heap, sizeof(*work));
	aligned_free(work_heap);

	return true;
}

static bool submit_work(struct thr_info *thr, const struct work *work_in)
{
	struct workio_cmd *wc;
	/* fill out work request message */
	wc = (struct workio_cmd *)calloc(1, sizeof(*wc));
	if (!wc)
		return false;

	wc->u.work = (struct work *)aligned_calloc(sizeof(*work_in));
	if (!wc->u.work)
		goto err_out;

	wc->cmd = WC_SUBMIT_WORK;
	wc->thr = thr;
	memcpy(wc->u.work, work_in, sizeof(*work_in));

	/* send solution to workio thread */
	if (!tq_push(thr_info[work_thr_id].q, wc))
		goto err_out;

	return true;

err_out:
	workio_cmd_free(wc);
	return false;
}

static void stratum_gen_work(struct stratum_ctx *sctx, struct work *work)
{
	uchar merkle_root[64];
	int i;

	if (!sctx->job.job_id) {
		// applog(LOG_WARNING, "stratum_gen_work: job not yet retrieved");
		return;
	}

	pthread_mutex_lock(&sctx->work_lock);

	// store the job ntime as high part of jobid
	snprintf(work->job_id, sizeof(work->job_id), "%07x %s",
		be32dec(sctx->job.ntime) & 0xfffffff, sctx->job.job_id);
	work->xnonce2_len = sctx->xnonce2_size;
	memcpy(work->xnonce2, sctx->job.xnonce2, sctx->xnonce2_size);

	// also store the bloc number
	work->height = sctx->job.height;

	/* Generate merkle root */
	switch (opt_algo) {
		case ALGO_HEAVY:
		case ALGO_MJOLLNIR:
			heavycoin_hash(merkle_root, sctx->job.coinbase, (int)sctx->job.coinbase_size);
			break;
		case ALGO_FUGUE256:
		case ALGO_GROESTL:
		case ALGO_KECCAK:
		case ALGO_BLAKECOIN:
			SHA256((uchar*)sctx->job.coinbase, sctx->job.coinbase_size, (uchar*)merkle_root);
			break;
		case ALGO_WHC:
		default:
			sha256d(merkle_root, sctx->job.coinbase, (int)sctx->job.coinbase_size);
	}

	for (i = 0; i < sctx->job.merkle_count; i++) {
		memcpy(merkle_root + 32, sctx->job.merkle[i], 32);
		if (opt_algo == ALGO_HEAVY || opt_algo == ALGO_MJOLLNIR)
			heavycoin_hash(merkle_root, merkle_root, 64);
		else
			sha256d(merkle_root, merkle_root, 64);
	}
	
//	/+Increment extranonce2 +/

	for (i = 0; i < (int)sctx->xnonce2_size && !++sctx->job.xnonce2[i]; i++);
	{
		sctx->job.xnonce2[i]++;		
	}



	/* Assemble block header */
	memset(work->data, 0, sizeof(work->data));
	work->data[0] = le32dec(sctx->job.version);
	for (i = 0; i < 8; i++)
		work->data[1 + i] = le32dec((uint32_t *)sctx->job.prevhash + i);
	for (i = 0; i < 8; i++)
		work->data[9 + i] = be32dec((uint32_t *)merkle_root + i);
	work->data[17] = le32dec(sctx->job.ntime);
	work->data[18] = le32dec(sctx->job.nbits);
	if (opt_algo == ALGO_MJOLLNIR || opt_algo == ALGO_HEAVY)
	{
		for (i = 0; i < 20; i++)
			work->data[i] = be32dec((uint32_t *)&work->data[i]);
	}

	work->data[20] = 0x80000000;
	work->data[31] = (opt_algo == ALGO_MJOLLNIR) ? 0x000002A0 : 0x00000280;

	// HeavyCoin (vote / reward)
	if (opt_algo == ALGO_HEAVY) {
		work->maxvote = 2048;
		uint16_t *ext = (uint16_t*)(&work->data[20]);
		ext[0] = opt_vote;
		ext[1] = be16dec(sctx->job.nreward);
		// applog(LOG_DEBUG, "DEBUG: vote=%hx reward=%hx", ext[0], ext[1]);
	}

	pthread_mutex_unlock(&sctx->work_lock);

	if (opt_debug) {
		char *tm = atime2str(swab32(work->data[17]) - sctx->srvtime_diff);
		char *xnonce2str = bin2hex(work->xnonce2, sctx->xnonce2_size);
		applog(LOG_DEBUG, "DEBUG: job_id=%s xnonce2=%s time=%s",
		       work->job_id, xnonce2str, tm);
		free(tm);
		free(xnonce2str);
	}

	switch (opt_algo) {
		case ALGO_JACKPOT:
		case ALGO_SCRYPT:
		case ALGO_SCRYPT_JANE:
		case ALGO_NEO:
			diff_to_target(work->target, sctx->job.diff / (65536.0 * opt_difficulty));
			break;
		case ALGO_DMD_GR:
		case ALGO_MYR_GR:
		case ALGO_FRESH:
		case ALGO_FUGUE256:
		case ALGO_GROESTL:
		case ALGO_KECCAK:
		case ALGO_LYRA2v2:
			diff_to_target(work->target, sctx->job.diff / (256.0 * opt_difficulty));
			break;
		case ALGO_LYRA2:
			diff_to_target(work->target, sctx->job.diff / (128.0 * opt_difficulty));
			break;
		default:
			diff_to_target(work->target, sctx->job.diff / opt_difficulty);
	}
}

static void restart_threads(void)
{
	if (opt_debug && !opt_quiet)
		applog(LOG_DEBUG,"%s", __FUNCTION__);
	for (int i = 0; i < opt_n_threads && work_restart; i++)
		work_restart[i].restart = 1;
}

static void *miner_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	int thr_id = mythr->id;
	struct work work;
	uint64_t loopcnt = 0;
	uint32_t max_nonce;
	uint32_t end_nonce = 0xffffffffU / opt_n_threads * (thr_id + 1) - (thr_id + 1);
	time_t firstwork_time = 0;
	bool work_done = false;
	bool extrajob = false;
	char s[16];
	int rc = 0;

	memset(&work, 0, sizeof(work)); // prevent work from being used uninitialized

	/* Set worker threads to nice 19 and then preferentially to SCHED_IDLE
	 * and if that fails, then SCHED_BATCH. No need for this to be an
	 * error if it fails */
	if (!opt_benchmark && opt_priority == 0) 
	{
		setpriority(PRIO_PROCESS, 0, 18);
		drop_policy();
	} else {
		int prio = 0;
#ifndef WIN32
		prio = -15;
		// note: different behavior on linux (-19 to 19)
		switch (opt_priority) {
			case 1:
				prio = 5;
				break;
			case 2:
				prio = 0;
				break;
			case 3:
				prio = -5;
				break;
			case 4:
				prio = -10;
				break;
			case 5:
				prio = -15;
		}
		applog(LOG_DEBUG, "Thread %d priority %d (set to %d)", thr_id,
			opt_priority, prio);
#endif
		int ret = setpriority(PRIO_PROCESS, 0, prio);
		if (opt_priority == 0) {
			drop_policy();
		}
	}

	/* Cpu thread affinity */
	if (num_cpus > 1) 
	{
		if (opt_affinity == -1) 
		{
			if (!opt_quiet)
				applog(LOG_DEBUG, "Binding thread %d to cpu %d (mask %x)", thr_id,
				thr_id%num_cpus, (1 << (thr_id)));
			affine_to_cpu_mask(thr_id, 1 << (thr_id));
		} else if (opt_affinity != -1) 
		{
			if (!opt_quiet)
				applog(LOG_DEBUG, "Binding thread %d to gpu mask %x", thr_id,
						opt_affinity);
			affine_to_cpu_mask(thr_id, opt_affinity);
		}
	}

	while (!abort_flag)
	{
		if (opt_benchmark)
		{
//			work.data[19] = work.data[19] & 0xfffffffU;	//reset Hashcounters
//			work.data[21] = work.data[21] & 0xfffffffU;
		}

		struct timeval tv_start, tv_end, diff;
		unsigned long hashes_done=0;
		uint32_t start_nonce;
		uint32_t scan_time = have_longpoll ? LP_SCANTIME : opt_scantime;
		uint64_t max64, minmax = 0x100000;

		// &work.data[19]
		int wcmplen = (opt_algo == ALGO_BITC) ? 140 : 76;
		uint32_t *nonceptr = (uint32_t*) (((char*)work.data) + wcmplen);

		if (have_stratum) 
		{
			uint32_t sleeptime = 0;
			while (!work_done && time(NULL) >= (g_work_time + 60)) 
			{
				usleep(100*1000);
				if (sleeptime > 4) {
					extrajob = true;
					break;
				}
				sleeptime++;
			}
			if (sleeptime && opt_debug && !opt_quiet)
			{
				applog(LOG_DEBUG, "sleeptime: %u ms", sleeptime * 100);
			}
				nonceptr = (uint32_t*) (((char*)work.data) + wcmplen);
			pthread_mutex_lock(&g_work_lock);
			extrajob |= work_done;
			if (nonceptr[0] >= end_nonce || extrajob) {
				work_done = false;
				extrajob = false;
				stratum_gen_work(&stratum, &g_work);
			}
		} else 
		{
			pthread_mutex_lock(&g_work_lock);
			if ((time(NULL) - g_work_time) >= scan_time || nonceptr[0] >= (end_nonce - 0x100)) {
				if (opt_debug && g_work_time && !opt_quiet)
					applog(LOG_DEBUG, "work time %u/%us nonce %x/%x", time(NULL) - g_work_time,
						scan_time, nonceptr[0], end_nonce);
				/* obtain new work from internal workio thread */
				if (unlikely(!get_work(mythr, &g_work))) {
					pthread_mutex_unlock(&g_work_lock);
					applog(LOG_ERR, "work retrieval failed, exiting mining thread %d", mythr->id);
					goto out;
				}
				g_work_time = time(NULL);
			}
		}

		if (!opt_benchmark && (g_work.height != work.height || memcmp(work.target, g_work.target, sizeof(work.target))))
		{
			calc_diff(&g_work, 0);
			if (!have_stratum)
				global_diff = g_work.difficulty;
			if (opt_debug) {
				uint64_t target64 = g_work.target[7] * 0x100000000ULL + g_work.target[6];
				applog(LOG_DEBUG, "job %s target change: %llx (%.1f)", g_work.job_id, target64, g_work.difficulty);
			}
			memcpy(work.target, g_work.target, sizeof(work.target));
			work.difficulty = g_work.difficulty;
			work.height = g_work.height;
			/* on new target, ignoring nonce, clear sent data (hashlog) */
			if (memcmp(work.target, g_work.target, sizeof(work.target))) {
				if (check_dups)
					hashlog_purge_job(work.job_id);
			}
		}
		if (memcmp(work.data, g_work.data, wcmplen)) {
			#if 0
			if (opt_debug) {
				for (int n=0; n <= (wcmplen-8); n+=8) {
					if (memcmp(work.data + n, g_work.data + n, 8)) {
						applog(LOG_DEBUG, "job %s work updated at offset %d:", g_work.job_id, n);
						applog_hash((uchar*) &work.data[n]);
						applog_compare_hash((uchar*) &g_work.data[n], (uchar*) &work.data[n]);
					}
				}
			}
			#endif
			memcpy(&work, &g_work, sizeof(struct work));
			nonceptr[0] = (UINT32_MAX / opt_n_threads) * thr_id; // 0 if single thr
		} else
			nonceptr[0]++; //??

		work_restart[thr_id].restart = 0;
		pthread_mutex_unlock(&g_work_lock);

		/* prevent gpu scans before a job is received */
		if ((have_stratum && work.data[0] == 0 || network_fail_flag) && !opt_benchmark)
		{
			sleep(1);
			continue;	
		}

		/* adjust max_nonce to meet target scan time */
		if (have_stratum)
			max64 = LP_SCANTIME;
		else
			max64 = max(1, scan_time + g_work_time - time(NULL));
		

		/* time limit */
		if (opt_time_limit && firstwork_time) {
			int passed = (int)(time(NULL) - firstwork_time);
			int remain = (int)(opt_time_limit - passed);
			if (remain < 0)  {
				abort_flag = true;
				if (opt_benchmark) {
					char rate[32];
					format_hashrate(global_hashrate, rate);
	                                applog(LOG_NOTICE, "Benchmark: %s", rate);
					usleep(200*1000);
					fprintf(stderr, "%llu\n", (long long unsigned int) global_hashrate);
				} else {
					applog(LOG_NOTICE, "Mining timeout of %ds reached, exiting...", opt_time_limit);
				}
				workio_abort();
				break;
			}
			if (remain < max64) max64 = remain;
		}


		max64 *= (uint32_t)thr_hashrates[thr_id];

		/* on start, max64 should not be 0,
		 *    before hashrate is computed */
		
		if (max64 < minmax) {
			switch (opt_algo) {
			case ALGO_KECCAK:
			case ALGO_BLAKECOIN:
			case ALGO_BLAKE:
			case ALGO_VANILLA:
			case ALGO_PENTABLAKE:
			case ALGO_WHC:
				minmax = 0x70000000U;
				break;
			case ALGO_SKEIN:
			case ALGO_BITCOIN:
			case ALGO_WHCX:
			case ALGO_QUBIT:
			case ALGO_QUARK:
			case ALGO_X14:
			case ALGO_LYRA2v2:
				minmax = 0x40000000U;
				break;
			case ALGO_DOOM:
			case ALGO_JACKPOT:
			case ALGO_LUFFA_DOOM:
			case ALGO_NEO:
				minmax = 0x2000000;
				break;
			case ALGO_X11:
			case ALGO_C11:
			case ALGO_X17:
			case ALGO_SCRYPT_JANE:
				minmax = 0x800000;
				break;
			case ALGO_S3:
				minmax = 0x400000;
				break;
			case ALGO_X13:
			case ALGO_X15:
			case ALGO_LYRA2:
				minmax = 0x300000;
				break;
			case ALGO_SCRYPT:
				minmax = 0x100000;
				break;
			}
			max64 = max(minmax-1, max64);
		}

		// we can't scan more than uint capacity
		max64 = min(UINT32_MAX, max64);
		start_nonce = nonceptr[0];

		if (opt_benchmark)
		{
			max_nonce = start_nonce + 0x5000000U;
		}
		else
		{

			/* never let small ranges at end */
			if (end_nonce >= UINT32_MAX - 256)
				end_nonce = UINT32_MAX;

			if ((max64 + start_nonce) >= end_nonce)
				max_nonce = end_nonce;
			else
				max_nonce = (uint32_t)(max64 + start_nonce);

			// todo: keep it rounded for gpu threads ?
			work.scanned_from = start_nonce;
			nonceptr[0] = start_nonce;
		}
		if (opt_debug)
			applog(LOG_DEBUG, "GPU #%d: start=%08x end=%08x range=%08x",
				device_map[thr_id], start_nonce, max_nonce, (max_nonce-start_nonce));

		hashes_done = 0;
		gettimeofday(&tv_start, NULL);

		/* scan nonces for a proof-of-work hash */
		switch (opt_algo) {

		case ALGO_HEAVY:
			rc = scanhash_heavy(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done, work.maxvote, HEAVYCOIN_BLKHDR_SZ);
			break;

		case ALGO_KECCAK:
			rc = scanhash_keccak256(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_MJOLLNIR:
			rc = scanhash_heavy(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done, 0, MNR_BLKHDR_SZ);
			break;

		case ALGO_DEEP:
			rc = scanhash_deep(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_DOOM:
		case ALGO_LUFFA_DOOM:
			rc = scanhash_doom(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_C11:
			rc = scanhash_c11(thr_id, work.data, work.target,
			max_nonce, &hashes_done);
			break;
				
		case ALGO_FUGUE256:
			rc = scanhash_fugue256(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_GROESTL:
		case ALGO_DMD_GR:
			rc = scanhash_groestlcoin(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_MYR_GR:
			rc = scanhash_myriad(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_JACKPOT:
			rc = scanhash_jackpot(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_QUARK:
			rc = scanhash_quark(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_QUBIT:
			rc = scanhash_qubit(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
		case ALGO_SCRYPT:
			rc = scanhash_scrypt(thr_id, work.data, work.target, NULL,
			max_nonce, &hashes_done, &tv_start, &tv_end);
			break;
			
		case ALGO_SCRYPT_JANE:
			rc = scanhash_scrypt_jane(thr_id, work.data, work.target, NULL,
			max_nonce, &hashes_done, &tv_start, &tv_end);
			break;

		case ALGO_SKEIN:
			rc = scanhash_skeincoin(thr_id, work.data, work.target,
				max_nonce, (uint32_t*)&hashes_done);
						break;

		case ALGO_BASTION:
			rc = scanhash_bastion(thr_id, work.data, work.target,
				max_nonce, &hashes_done);
			break;

		case ALGO_BITCOIN:
			rc = scanhash_bitcoin(thr_id, work.data, work.target,
				max_nonce, &hashes_done);
			break;

		case ALGO_BLAKECOIN:
			rc = scanhash_blake256(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done, 8);
			break;

		case ALGO_BLAKE:
			rc = scanhash_blake256(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done, 14);
			break;

		case ALGO_FRESH:
			rc = scanhash_fresh(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_LYRA2:
			rc = scanhash_lyra2(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;
		case ALGO_LYRA2v2:
			rc = scanhash_lyra2v2(thr_id, work.data, work.target,
				max_nonce, &hashes_done);
			break;

		case ALGO_NIST5:
			rc = scanhash_nist5(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_PENTABLAKE:
			rc = scanhash_pentablake(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_S3:
			rc = scanhash_s3(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_WHC:
			rc = scanhash_whc(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_WHCX:
			rc = scanhash_whirlpoolx(thr_id, work.data, work.target,
				max_nonce, (uint32_t*)&hashes_done);
			break;

		case ALGO_X11:
			rc = scanhash_x11(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_X13:
			rc = scanhash_x13(thr_id, work.data, work.target,
			                      max_nonce, &hashes_done);
			break;

		case ALGO_X14:
			rc = scanhash_x14(thr_id, work.data, work.target,
				max_nonce, &hashes_done);
			break;

		case ALGO_X15:
			rc = scanhash_x15(thr_id, work.data, work.target,
				max_nonce, &hashes_done);
			break;

		case ALGO_X17:
			rc = scanhash_x17(thr_id, work.data, work.target,
				max_nonce, &hashes_done);
			break;
		case ALGO_VANILLA:
			rc = scanhash_blake256(thr_id, work.data, work.target,
				max_nonce, &hashes_done, 8);
			break;

		case ALGO_NEO:
			rc = scanhash_neoscrypt(have_stratum, thr_id, work.data, work.target, max_nonce, &hashes_done);
			break;
		case ALGO_BITC:
			rc = scanhash_bitcredit(thr_id, work.data, work.target, work.midstate, max_nonce, &hashes_done);
			break;			
		default:
			/* should never happen */
			goto out;
		}

		/* record scanhash elapsed time */
		gettimeofday(&tv_end, NULL);

		if (firstwork_time == 0)
			firstwork_time = time(NULL);

		if (rc && opt_debug)
			applog(LOG_NOTICE, CL_CYN "found => %08x" CL_GRN " %08x", nonceptr[0], swab32(nonceptr[0])); // data[19]
		if (rc > 1 && opt_debug)
			applog(LOG_NOTICE, CL_CYN "found => %08x" CL_GRN " %08x", nonceptr[2], swab32(nonceptr[2])); // data[21]

		timeval_subtract(&diff, &tv_end, &tv_start);

//		diff.tv_sec == 0 &&
		if (diff.tv_sec > 0 || (diff.tv_sec == 0 && diff.tv_usec>2000)) // avoid totally wrong hash rates
		{
			double dtime = (double) diff.tv_sec + 1e-6 * diff.tv_usec;

			/* hashrate factors for some algos */
			double rate_factor = 1.0;
			switch (opt_algo) {
				case ALGO_JACKPOT:
				case ALGO_QUARK:
					// to stay comparable to other ccminer forks or pools
					rate_factor = 0.5;
					break;
			}

			/* store thread hashrate */
			if (dtime > 0.0) {
				pthread_mutex_lock(&stats_lock);
				thr_hashrates[thr_id] = hashes_done / dtime;
				thr_hashrates[thr_id] *= rate_factor;
				stats_remember_speed(thr_id, hashes_done, thr_hashrates[thr_id], (uint8_t) rc, work.height);
				pthread_mutex_unlock(&stats_lock);
			}
		}

		work.scanned_to = start_nonce + hashes_done;
		if (opt_debug && opt_benchmark) 
		{
			// to debug nonce ranges
			applog(LOG_DEBUG, "GPU #%d:  ends=%08x range=%llx", device_map[thr_id],
				start_nonce + hashes_done, hashes_done);
		}

		if (check_dups)
			hashlog_remember_scan_range(&work);

		if (((opt_algo == ALGO_BITC) ? (loopcnt % 400 == 0) : (loopcnt)))
		{
			bool   writelog = false;
			double hashrate = 0.0;

			if (opt_n_gputhreads != 1)
			{
				if (loopcnt%opt_n_gputhreads == 0 ) //Display the hash 1 time per gpu and not opt_n_gputhreads times per gpu
				{
					int index = thr_id / opt_n_gputhreads;
					for (int i = 0; i < opt_n_gputhreads; i++)
					{
						hashrate += thr_hashrates[(index*opt_n_gputhreads) + i];
					}
					if (!opt_quiet) writelog = true;
				}
			}
			else
			{	

				if(!opt_quiet) writelog = true;
				hashrate = thr_hashrates[thr_id];
			}
			if (hashrate == 0.0) writelog = false;
			if (writelog)
			{
#ifdef USE_WRAPNVML
				if (hnvml != NULL) {
					uint32_t tempC=0, fanpcnt=0, mwatts=0, graphics_clock=0, mem_clock=0;

					nvml_get_tempC(hnvml, device_map[thr_id], &tempC);
					nvml_get_fanpcnt(hnvml, device_map[thr_id], &fanpcnt);
					nvml_get_current_clocks(hnvml, device_map[thr_id], &graphics_clock, &mem_clock);
					//if (nvml_get_power_usage(hnvml, device_map[thr_id], &mwatts) == 0)
					//    sprintf(gpupowbuf, "%dW", (mwatts / 1000));

					applog(LOG_INFO, "GPU #%d: %s, %*.f (T=%3dC F=%3d%% C=%d/%d)", device_map[thr_id], device_name[device_map[thr_id]], (hashrate > 1e6) ? 0 : 2, 1e-3 * hashrate, tempC, fanpcnt, graphics_clock, mem_clock);
				}
				else
#endif
				{
					applog(LOG_INFO, "GPU #%d: %s, %*.f", device_map[thr_id], device_name[device_map[thr_id]], (hashrate > 1e6) ? 0 : 2, 1e-3 * hashrate);
				}
			}
		}

		/* loopcnt: ignore first loop hashrate */
		if ((loopcnt>0) && thr_id == (opt_n_threads - 1)) 
		{
			double hashrate = 0.;
			pthread_mutex_lock(&stats_lock);
			for (int i = 0; i < opt_n_threads; i++)
				hashrate += stats_get_speed(i, thr_hashrates[i]);
			pthread_mutex_unlock(&stats_lock);
			if (opt_benchmark) 
			{
				double hashrate = 0.;
				pthread_mutex_lock(&stats_lock);
				for (int i = 0; i < opt_n_threads && thr_hashrates[i]; i++)
					hashrate += stats_get_speed(i, thr_hashrates[i]);
				pthread_mutex_unlock(&stats_lock);
				if (opt_benchmark && loopcnt >1) {
					format_hashrate(hashrate, s);
					applog(LOG_NOTICE, "Total: %s", s);
				}
				// X-Mining-Hashrate
				global_hashrate = llround(hashrate);	
			}
		}

		/* if nonce found, submit work */
		if (rc && !opt_benchmark) {
			if (!submit_work(mythr, &work))
				break;

			// prevent stale work in solo
			// we can't submit twice a block!
			if (!have_stratum) {
				pthread_mutex_lock(&g_work_lock);
				// will force getwork
				g_work_time = 0;
				pthread_mutex_unlock(&g_work_lock);
				continue;
			}

			// second nonce found, submit too (on pool only!)
			if (rc > 1 && work.data[21]) {
				work.data[19] = work.data[21];
				work.data[21] = 0;
				if (!submit_work(mythr, &work))
					break;
			}
		}
		work.data[19] = start_nonce + hashes_done;
		loopcnt++;
	}

	return NULL;

out:
	tq_freeze(mythr->q);

	return NULL;
}

static void *longpoll_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	CURL *curl = NULL;
	char *copy_start, *hdr_path = NULL, *lp_url = NULL;
	bool need_slash = false;

	curl = curl_easy_init();
	if (unlikely(!curl)) {
		applog(LOG_ERR, "CURL initialization failed");
		goto out;
	}

	hdr_path = (char*)tq_pop(mythr->q, NULL);
	if (!hdr_path)
		goto out;

	/* full URL */
	if (strstr(hdr_path, "://")) {
		lp_url = hdr_path;
		hdr_path = NULL;
	}
	
	/* absolute path, on current server */
	else {
		copy_start = (*hdr_path == '/') ? (hdr_path + 1) : hdr_path;
		if (rpc_url[strlen(rpc_url) - 1] != '/')
			need_slash = true;

		lp_url = (char*)malloc(strlen(rpc_url) + strlen(copy_start) + 2);
		if (!lp_url)
			goto out;

		sprintf(lp_url, "%s%s%s", rpc_url, need_slash ? "/" : "", copy_start);
	}

	applog(LOG_INFO, "Long-polling activated for %s", lp_url);

	while (!abort_flag) {
		json_t *val, *soval;
		int err;

		val = json_rpc_call(curl, lp_url, rpc_userpass, rpc_req,
				    false, true, &err);


		network_fail_flag = (err != CURLE_OK);

		if (have_stratum) {
			if (val)
				json_decref(val);
			goto out;
		}
		if (likely(val)) 
		{
			if (!opt_quiet) applog(LOG_INFO, "LONGPOLL detected new block");
			soval = json_object_get(json_object_get(val, "result"), "submitold");
			submit_old = soval ? json_is_true(soval) : false;
			pthread_mutex_lock(&g_work_lock);
			if (work_decode(json_object_get(val, "result"), &g_work)) {
				if (opt_debug)
					applog(LOG_BLUE, "LONGPOLL pushed new work");
				g_work_time = time(NULL);
				applog(LOG_BLUE, "Restart threafds");
				restart_threads();
			}
			pthread_mutex_unlock(&g_work_lock);
			json_decref(val);
		} else {
			pthread_mutex_lock(&g_work_lock);
			g_work_time -= LP_SCANTIME;
			pthread_mutex_unlock(&g_work_lock);
			restart_threads();
			if (err != CURLE_OPERATION_TIMEDOUT) {
				have_longpoll = false;
				free(hdr_path);
				free(lp_url);
				lp_url = NULL;
				sleep(opt_fail_pause);
			}
		}
	}

out:
	free(hdr_path);
	free(lp_url);
	tq_freeze(mythr->q);
	if (curl)
		curl_easy_cleanup(curl);

	return NULL;
}

static bool stratum_handle_response(char *buf)
{
	json_t *val, *err_val, *res_val, *id_val;
	json_error_t err;
	struct timeval tv_answer, diff;
	bool ret = false;

	val = JSON_LOADS(buf, &err);
	if (!val) {
		applog(LOG_INFO, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");
	id_val = json_object_get(val, "id");

	if (!id_val || json_is_null(id_val) || !res_val)
		goto out;

	// ignore subscribe late answer (yaamp)
	if (json_integer_value(id_val) < 4)
		goto out;

	gettimeofday(&tv_answer, NULL);
	timeval_subtract(&diff, &tv_answer, &stratum.tv_submit);
	// store time required to the pool to answer to a submit
	stratum.answer_msec = (1000 * diff.tv_sec) + (uint32_t) (0.001 * diff.tv_usec);

	share_result(json_is_true(res_val),
		err_val ? json_string_value(json_array_get(err_val, 1)) : NULL);

	ret = true;
out:
	if (val)
		json_decref(val);

	return ret;
}

static void *stratum_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	char *s;

	stratum.url = (char*)tq_pop(mythr->q, NULL);
	if (!stratum.url)
		goto out;
	applog(LOG_BLUE, "Starting Stratum on %s", stratum.url);

	while (!abort_flag) {
		int failures = 0;

		if (stratum_need_reset) {
			stratum_need_reset = false;
			stratum_disconnect(&stratum);
			applog(LOG_DEBUG, "stratum connection reset");
		}

		while (!stratum.curl && !abort_flag) {
			pthread_mutex_lock(&g_work_lock);
			g_work_time = 0;
			pthread_mutex_unlock(&g_work_lock);
			restart_threads();

			if (!stratum_connect(&stratum, stratum.url) ||
			    !stratum_subscribe(&stratum) ||
			    !stratum_authorize(&stratum, rpc_user, rpc_pass,opt_extranonce)) {
				stratum_disconnect(&stratum);
				network_fail_flag = true;

				if (opt_retries >= 0 && ++failures > opt_retries) {
					applog(LOG_ERR, "...terminating workio thread");
					tq_push(thr_info[work_thr_id].q, NULL);
					abort_flag = true;
					goto out;
				}
				if (!opt_benchmark)
					applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);
				sleep(opt_fail_pause);
			}
		}

		if (stratum.job.job_id &&
		    (!g_work_time || strncmp(stratum.job.job_id, g_work.job_id + 8, 120))) {
			pthread_mutex_lock(&g_work_lock);
			stratum_gen_work(&stratum, &g_work);
			g_work_time = time(NULL);
			if (stratum.job.clean) 
			{
				network_fail_flag = false;
				if (!opt_quiet)
					applog(LOG_BLUE, "%s %s block %d", short_url, algo_names[opt_algo],
						stratum.job.height);
				restart_threads();
				if (check_dups)
					hashlog_purge_old();
				stats_purge_old();
			} else if (opt_debug && !opt_quiet) {
					applog(LOG_BLUE, "%s asks job %d for block %d", short_url,
						strtoul(stratum.job.job_id, NULL, 16), stratum.job.height);
			}
			pthread_mutex_unlock(&g_work_lock);
		}
		
		if (!stratum_socket_full(&stratum, 120)) {
			applog(LOG_ERR, "Stratum connection timed out");
			s = NULL;
		} else
			s = stratum_recv_line(&stratum);
		if (!s) {
			stratum_disconnect(&stratum);
			applog(LOG_ERR, "Stratum connection interrupted");
			continue;
		}
		if (!stratum_handle_method(&stratum, s))
			stratum_handle_response(s);
		free(s);
	}

	stratum_disconnect(&stratum);

out:
	return NULL;
}

static void show_version_and_exit(void)
{
	printf("%s v%s\n"
#ifdef WIN32
		"pthreads static %s\n"
#endif
		"%s\n",
		PACKAGE_NAME, PACKAGE_VERSION,
#ifdef WIN32
		PTW32_VERSION_STRING,
#endif
		curl_version());
	proper_exit(0);
}

static void show_usage_and_exit(int status)
{
	if (status)
		fprintf(stderr, "Try `" PROGRAM_NAME " --help' for more information.\n");
	else
		printf(usage);
	if (opt_algo == ALGO_SCRYPT || opt_algo == ALGO_SCRYPT_JANE) 
	{
		printf(scrypt_usage);	
	}
	proper_exit(0);
}

static void parse_arg(int key, char *arg)
{
	char *p = arg;
	int v, i;
	double d;
	char *pch;
	int n;
	int last;
	opterr = 1;

	switch(key) {
	case 'C':
		opt_cpumining = true;
		applog(LOG_INFO, "Cpu mining enabled...");
		break;
	case 'a':
		p = strstr(arg, ":"); // optional factor
		if (p) *p = '\0';
		for (i = 0; i < ALGO_COUNT; i++) {
			if (algo_names[i] && !strcasecmp(arg, algo_names[i])) {
				opt_algo = (enum sha_algos)i;
				break;
			}
		}
		if (i == ALGO_COUNT)
			show_usage_and_exit(1);

		if (p) 
		{
			opt_nfactor = atoi(p + 1);
			if (opt_algo == ALGO_SCRYPT_JANE) 
			{
				free(jane_params);
				jane_params = strdup(p + 1);				
			}			
		}
		if (!opt_nfactor) 
		{
			switch (opt_algo) 
			{
				case ALGO_SCRYPT:      opt_nfactor = 9;  break;
				case ALGO_SCRYPT_JANE: opt_nfactor = 14; break;				
			}
		}
		break;
	case 'b':
		p = strstr(arg, ":");
		if (p) {
			/* ip:port */
			if (p - arg > 0) {
				free(opt_api_allow);
				opt_api_allow = strdup(arg);
				opt_api_allow[p - arg] = '\0';
			}
			opt_api_listen = atoi(p + 1);
		}
		else if (arg && strstr(arg, ".")) {
			/* ip only */
			free(opt_api_allow);
			opt_api_allow = strdup(arg);
		}
		else if (arg) {
			/* port or 0 to disable */
			opt_api_listen = atoi(arg);
		}
		break;
	case 'B':
		opt_background = true;
		break;
	case 'c': {
		json_error_t err;
		if (opt_config)
			json_decref(opt_config);
#if JANSSON_VERSION_HEX >= 0x020000
		opt_config = json_load_file(arg, 0, &err);
#else
		opt_config = json_load_file(arg, &err);
#endif
		if (!json_is_object(opt_config)) {
			applog(LOG_ERR, "JSON decode of %s failed", arg);
			proper_exit(1);
		}
		break;
	}
	case 'i':
		d = atof(arg);
		v = (uint32_t) d;
		if (v < 0 || v > 31)
			show_usage_and_exit(1);
		{
			int n = 0;
			int ngpus = cuda_num_devices();
			uint32_t last = 0;
			char * pch = strtok(arg,",");
			while (pch != NULL) {
				d = atof(pch);
				v = (uint32_t) d;
				if (v > 7) { /* 0 = default */
					if ((d - v) > 0.0) {
						uint32_t adds = (uint32_t)floor((d - v) * (1 << (v - 8))) * 256;
						gpus_intensity[n] = (1 << v) + adds;
						applog(LOG_INFO, "Adding %u threads to intensity %u, %u cuda threads",
							adds, v, gpus_intensity[n]);
					}
					else if (gpus_intensity[n] != (1 << v)) {
						gpus_intensity[n] = (1 << v);
						applog(LOG_INFO, "Intensity set to %u, %u cuda threads",
							v, gpus_intensity[n]);
					}
				}
				last = gpus_intensity[n];
				n++;
				pch = strtok(NULL, ",");
			}
			while (n < MAX_GPUS)
				gpus_intensity[n++] = last;
		}
		break;
	case 'X':
		v = atoi(arg);
		if (v < 0 || v > 999)
			show_usage_and_exit(1);
		{
			int n = 0;
			int ngpus = cuda_num_devices();
			uint32_t last = 0;
			char * pch = strtok(arg,",");
			while (pch != NULL) {
				d = atof(pch);
				v = (uint32_t) d;
					if ((d - v) > 0) {
						uint32_t adds = (v*256*256);
						gpus_intensity[n] = (1 << v) + adds;
						applog(LOG_INFO, "Adding %u threads to Xintensity %u, %u cuda threads",
							adds, v, gpus_intensity[n]);
					}
					else if (gpus_intensity[n] != (1 << v)) {
						gpus_intensity[n] = (v*256*256);
						applog(LOG_INFO, "XIntensity set to %u, %u cuda threads",
							v, gpus_intensity[n]);
					}
				last = gpus_intensity[n];
				n++;
				pch = strtok(NULL, ",");
			}
			while (n < MAX_GPUS)
				gpus_intensity[n++] = last;
		}
		break;
	case 'n': /* --ndevs */
		cuda_print_devices();
		proper_exit(0);
		break;
	case 'N':
		v = atoi(arg);
		if (v < 1)
			opt_statsavg = INT_MAX;
		opt_statsavg = v;
		break;
	case 'q':
		opt_quiet = true;
		break;
	case 'p':
		free(rpc_pass);
		rpc_pass = strdup(arg);
		break;
	case 'P':
		opt_protocol = true;
		break;
	case 'r':
		v = atoi(arg);
		if (v < -1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_retries = v;
		break;
	case 'R':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_fail_pause = v;
		break;
	case 's':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_scantime = v;
		break;
	case 'T':
		v = atoi(arg);
		if (v < 1 || v > 99999)	/* sanity check */
			show_usage_and_exit(1);
		opt_timeout = v;
		break;
	case 't':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_n_threads = v;
		break;
	case 'v':
		v = atoi(arg);
		if (v < 0 || v > 8192)	/* sanity check */
			show_usage_and_exit(1);
		opt_vote = (uint16_t)v;
		break;
	case 'm':
		opt_trust_pool = true;
		break;
	case 'u':
		free(rpc_user);
		rpc_user = strdup(arg);
		break;
	case 'o':			/* --url */
		p = strstr(arg, "://");
		if (p) {
			if (strncasecmp(arg, "http://", 7) && strncasecmp(arg, "https://", 8) &&
					strncasecmp(arg, "stratum+tcp://", 14))
				show_usage_and_exit(1);
			free(rpc_url);
			rpc_url = strdup(arg);
			short_url = &rpc_url[(p - arg) + 3];
		} else {
			if (!strlen(arg) || *arg == '/')
				show_usage_and_exit(1);
			free(rpc_url);
			rpc_url = (char*)malloc(strlen(arg) + 8);
			sprintf(rpc_url, "http://%s", arg);
			short_url = &rpc_url[7];
		}
		p = strrchr(rpc_url, '@');
		if (p) {
			char *sp, *ap;
			*p = '\0';
			ap = strstr(rpc_url, "://") + 3;
			sp = strchr(ap, ':');
			if (sp) {
				free(rpc_userpass);
				rpc_userpass = strdup(ap);
				free(rpc_user);
				rpc_user = (char*)calloc(sp - ap + 1, 1);
				strncpy(rpc_user, ap, sp - ap);
				free(rpc_pass);
				rpc_pass = strdup(sp + 1);
			} else {
				free(rpc_user);
				rpc_user = strdup(ap);
			}
			memmove(ap, p + 1, strlen(p + 1) + 1);
			short_url = p + 1;
		}
		have_stratum = !opt_benchmark && !strncasecmp(rpc_url, "stratum", 7);
		break;
	case 'O':			/* --userpass */
		p = strchr(arg, ':');
		if (!p)
			show_usage_and_exit(1);
		free(rpc_userpass);
		rpc_userpass = strdup(arg);
		free(rpc_user);
		rpc_user = (char*)calloc(p - arg + 1, 1);
		strncpy(rpc_user, arg, p - arg);
		free(rpc_pass);
		rpc_pass = strdup(p + 1);
		break;
	case 'x':			/* --proxy */
		if (!strncasecmp(arg, "socks4://", 9))
			opt_proxy_type = CURLPROXY_SOCKS4;
		else if (!strncasecmp(arg, "socks5://", 9))
			opt_proxy_type = CURLPROXY_SOCKS5;
#if LIBCURL_VERSION_NUM >= 0x071200
		else if (!strncasecmp(arg, "socks4a://", 10))
			opt_proxy_type = CURLPROXY_SOCKS4A;
		else if (!strncasecmp(arg, "socks5h://", 10))
			opt_proxy_type = CURLPROXY_SOCKS5_HOSTNAME;
#endif
		else
			opt_proxy_type = CURLPROXY_HTTP;
		free(opt_proxy);
		opt_proxy = strdup(arg);
		break;
	case 1001:
		free(opt_cert);
		opt_cert = strdup(arg);
		break;
	case 1002:
		use_colors = false;
		break;
	case 1004:
		opt_autotune = false;
		break;
		case 'l': /* scrypt --launch-config */
		{
				char *last = NULL, *pch = strtok(arg, ",");
				int n = 0;
				while (pch != NULL) {
					device_config[n++] = last = strdup(pch);
					pch = strtok(NULL, ",");
					
				}
				while (n < MAX_GPUS)
				device_config[n++] = last;
		}
		break;
		case 'L': /* scrypt --lookup-gap */
		{
			char *pch = strtok(arg, ",");
			int n = 0, last = 0;
			while (pch != NULL) 
			{
				device_lookup_gap[n++] = last = atoi(pch);
				pch = strtok(NULL, ",");				
			}
			while (n < MAX_GPUS)
				 device_lookup_gap[n++] = last;
		}
		break;
	case 1005:
		opt_benchmark = true;
		want_longpoll = false;
		want_stratum = false;
		have_stratum = false;
		break;
	case 1006:
		print_hash_tests();
		proper_exit(0);
		break;
	case 1003:
		want_longpoll = false;
		break;
	case 1007:
		want_stratum = false;
		break;
	case 1008:
		opt_time_limit = atoi(arg);
		break;
	case 1011:
		allow_gbt = false;
		break;
	case 'S':
	case 1018:
		applog(LOG_INFO, "Now logging to syslog...");
		use_syslog = true;
		if (arg && strlen(arg)) {
			free(opt_syslog_pfx);
			opt_syslog_pfx = strdup(arg);
		}
		break;
	case 1020:
		v = atoi(arg);
		if (v < -1)
			v = -1;
		if (v > (1<<num_cpus)-1)
			v = -1;
		opt_affinity = v;
		break;
	case 1021:
		v = atoi(arg);
		if (v < 0 || v > 5)	/* sanity check */
			show_usage_and_exit(1);
		opt_priority = v;
		break;
	case 'd': // CB
		{
			int ngpus = cuda_num_devices();
			char * pch = strtok (arg,",");
			opt_n_threads = 0;
			while (pch != NULL) {
				if (pch[0] >= '0' && pch[0] <= '9' && pch[1] == '\0')
				{
					if (atoi(pch) < ngpus)
						device_map[opt_n_threads++] = atoi(pch);
					else {
						applog(LOG_ERR, "Non-existant CUDA device #%d specified in -d option", atoi(pch));
						proper_exit(1);
					}
				} else {
					int device = cuda_finddevice(pch);
					if (device >= 0 && device < ngpus)
						device_map[opt_n_threads++] = device;
					else {
						applog(LOG_ERR, "Non-existant CUDA device '%s' specified in -d option", pch);
						proper_exit(1);
					}
				}
				// set number of active gpus
				active_gpus = opt_n_threads;
				pch = strtok (NULL, ",");
			}
		}
		break;
	case 'f': // CH - Divisor for Difficulty
		d = atof(arg);
		if (d == 0)	/* sanity check */
			show_usage_and_exit(1);
		opt_difficulty = d;
		break;

	case 'e':
		opt_extranonce = false;
		break;
	case 1070:
		pch = strtok(arg, ",");
		n = 0, last = atoi(arg);
		while (pch != NULL)
		{
			device_gpuspeed[n++] = last = atoi(pch);
			pch = strtok(NULL, ",");
		}
		break;
	case 1071:
		pch = strtok(arg, ",");
		n = 0, last = atoi(arg);
		while (pch != NULL) 
		{
			device_memspeed[n++] = last = atoi(pch);
			pch = strtok(NULL, ",");		
		}
		break;
	case 'g':
		v = atoi(arg);
		if (v < 1 || v > 9999)	/* sanity check */
			show_usage_and_exit(1);
		opt_n_gputhreads = v;

		int buf[MAX_GPUS];
		for (int i = 0; i < active_gpus; i++)
		{
			buf[i] = device_map[i];
		}
		for (int i = 0; i < active_gpus; i++)
		{
			for (int j = 0; j<opt_n_gputhreads; j++)
			{
				device_map[(i * opt_n_gputhreads) + j] = buf[i];
			}
		}
		opt_n_threads = active_gpus*opt_n_gputhreads;
		active_gpus= opt_n_threads;
		opt_extranonce = false;
		break;
	case 1030:
		opt_broken_neo_wallet = true;
		break;
	case 1080:
		opt_eco_mode = true;
		break;
	case 'D':
		opt_debug = true;
		break;
	case 'V':
		show_version_and_exit();
	case 'h':
		show_usage_and_exit(0);
	default:
		show_usage_and_exit(1);
	}

	if (use_syslog)
		use_colors = false;
}


/**
 * Parse json config file
 */
static void parse_config(void)
{
	int i;
	json_t *val;

	if (!json_is_object(opt_config))
		return;

	for (i = 0; i < ARRAY_SIZE(options); i++) {

		if (!options[i].name)
			break;
		if (!strcmp(options[i].name, "config"))
			continue;

		val = json_object_get(opt_config, options[i].name);
		if (!val)
			continue;

		if (options[i].has_arg && json_is_string(val)) {
			char *s = strdup(json_string_value(val));
			if (!s)
				continue;
			parse_arg(options[i].val, s);
			free(s);
		}
		else if (options[i].has_arg && json_is_integer(val)) {
			char buf[16];
			sprintf(buf, "%d", (int) json_integer_value(val));
			parse_arg(options[i].val, buf);
		}
		else if (options[i].has_arg && json_is_real(val)) {
			char buf[16];
			sprintf(buf, "%f", json_real_value(val));
			parse_arg(options[i].val, buf);
		}
		else if (!options[i].has_arg) {
			if (json_is_true(val))
				parse_arg(options[i].val, (char*) "");
		}
		else
			applog(LOG_ERR, "JSON option %s invalid",
				options[i].name);
	}
}

static void parse_cmdline(int argc, char *argv[])
{
	int key;

	while (1) {
#if HAVE_GETOPT_LONG
		key = getopt_long(argc, argv, short_options, options, NULL);
#else
		key = getopt(argc, argv, short_options);
#endif
		if (key < 0)
			break;

		parse_arg(key, optarg);
	}
	if (optind < argc) {
		fprintf(stderr, "%s: unsupported non-option argument '%s'\n",
			argv[0], argv[optind]);
		show_usage_and_exit(1);
	}

	parse_config();

	if (opt_algo == ALGO_HEAVY && opt_vote == 9999) {
		fprintf(stderr, "%s: Heavycoin hash requires block reward vote parameter (see --vote)\n",
			argv[0]);
		show_usage_and_exit(1);
	}
}

#ifndef WIN32
static void signal_handler2(int sig)
{
	switch (sig) {
	case SIGINT:
		signal(sig, SIG_IGN);
		applog(LOG_INFO, "SIGINT received, aborting miner jobs");
                scan_abort_flag = true;
		break;
	}
}
static void signal_handler(int sig)
{
	switch (sig) {
	case SIGHUP:
		applog(LOG_INFO, "SIGHUP received");
		break;
	case SIGINT:
		signal(sig, signal_handler2);
		applog(LOG_INFO, "SIGINT received, exiting once miner jobs complete.  Ctrl+C again to abort miner jobs");
		proper_exit(CCEXIT_SIG);
		break;
	case SIGTERM:
		scan_abort_flag = true;
		applog(LOG_INFO, "SIGTERM received, exiting");
		proper_exit(CCEXIT_SIG);
		break;
	}
}
#else
BOOL WINAPI ConsoleHandler(DWORD dwType)
{
	switch (dwType) {
	case CTRL_C_EVENT:
	{
		static bool called = false;
		if (!called) {
			called = true;
			applog(LOG_INFO, "CTRL_C_EVENT received, exiting once miner jobs complete.  Ctrl+C again to abort miner jobs");
			proper_exit(CCEXIT_SIG);
		} else {
			applog(LOG_INFO, "CTRL_C_EVENT received, aborting miner jobs");
			scan_abort_flag = true;
		}


		break;
	}
	case CTRL_BREAK_EVENT:
		applog(LOG_INFO, "CTRL_BREAK_EVENT received, exiting");
		proper_exit(CCEXIT_SIG);
		break;
	default:
		return false;
	}
	return true;
}
#endif

int main(int argc, char *argv[])
{
	struct thr_info *thr;
	long flags;
	int i;
	
	// strdup on char* to allow a common free() if used
	opt_syslog_pfx = strdup(PROGRAM_NAME);
	opt_api_allow = strdup("127.0.0.1"); /* 0.0.0.0 for all ips */

	printf("SP-Mod 1.5.80 \n");
#ifdef _MSC_VER
	printf("Compiled with Visual C++ %d ", _MSC_VER / 100);
#else
#ifdef __clang__
	printf("Compiled with Clang %s ", __clang_version__);
#else
#ifdef __GNUC__
	printf("Compiled with GCC %d.%d ", __GNUC__, __GNUC_MINOR__);
#else
	printf("Compiled with an unknown compiler ");
#endif
#endif
#endif
	printf("using Nvidia CUDA Toolkit %d.%d\n\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
	printf("  Based on pooler cpuminer 2.3.2 and the tpruvot@github fork\n");
	printf("  CUDA support by Christian Buchner, Christian H. and DJM34\n");
	printf("  Includes optimizations implemented by sp, klaust, tpruvot and tsiv.\n\n");

	rpc_user = strdup("");
	rpc_pass = strdup("");
	jane_params = strdup("");

	// number of cpus for thread affinity
#if defined(WIN32)
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	num_cpus = sysinfo.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_CONF)
	num_cpus = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(CTL_HW) && defined(HW_NCPU)
	int req[] = { CTL_HW, HW_NCPU };
	size_t len = sizeof(num_cpus);
	sysc tl(req, 2, &num_cpus, &len, NULL, 0);
#else
	num_cpus = 1;
#endif
	// number of gpus
	active_gpus = cuda_num_devices();


	if (active_gpus > 1)
	{
		// default thread to device map
		for (i = 0; i < MAX_GPUS; i++)
		{
			device_map[i] = i;
			device_name[i] = NULL;
					// for future use, maybe
			device_interactive[i] = -1;
			device_batchsize[i] = 1024;
			device_backoff[i] = is_windows() ? 12 : 2;
			device_lookup_gap[i] = 1;
			device_texturecache[i] = -1;
			device_singlememory[i] = -1;
			device_config[i] = NULL;
		}
	}

	cuda_devicenames();

	/* parse command line */
	parse_cmdline(argc, argv);
	if (abort_flag) return 0;

	if (!opt_benchmark && !rpc_url) {
		fprintf(stderr, "%s: no URL supplied\n", argv[0]);
		show_usage_and_exit(1);
	}
		
	if (!rpc_userpass) {
		rpc_userpass = (char*)malloc(strlen(rpc_user) + strlen(rpc_pass) + 2);
		if (!rpc_userpass)
			return 1;
		sprintf(rpc_userpass, "%s:%s", rpc_user, rpc_pass);
	}


	cuda_devicereset();

	/* init stratum data.. */
	memset(&stratum.url, 0, sizeof(stratum));

	pthread_mutex_init(&stratum.sock_lock, NULL);
	pthread_mutex_init(&stratum.work_lock, NULL);

	flags = !opt_benchmark && rpc_url && strncmp(rpc_url, "https:", 6)
	      ? (CURL_GLOBAL_ALL & ~CURL_GLOBAL_SSL)
	      : CURL_GLOBAL_ALL;
	if (curl_global_init(flags)) {
		applog(LOG_ERR, "CURL initialization failed");
		return 1;
	}

#ifndef WIN32
	if (opt_background) {
		i = fork();
		if (i < 0) exit(1);
		if (i > 0) exit(0);
		i = setsid();
		if (i < 0)
			applog(LOG_ERR, "setsid() failed (errno = %d)", errno);
		i = chdir("/");
		if (i < 0)
			applog(LOG_ERR, "chdir() failed (errno = %d)", errno);
		signal(SIGHUP, signal_handler);
		signal(SIGTERM, signal_handler);
	}
	/* Always catch Ctrl+C */
	signal(SIGINT, signal_handler);
#else
	SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE);
	if (opt_priority > 0)
	{
		DWORD prio = NORMAL_PRIORITY_CLASS;
		prio = REALTIME_PRIORITY_CLASS;//default realtime

		switch (opt_priority) {
		case 1:
			prio = BELOW_NORMAL_PRIORITY_CLASS;
			break;
		case 3:
			prio = ABOVE_NORMAL_PRIORITY_CLASS;
			break;
		case 4:
			prio = HIGH_PRIORITY_CLASS;
			break;
		case 5:
			prio = REALTIME_PRIORITY_CLASS;
		}
		if (SetPriorityClass(GetCurrentProcess(), prio) == 0)
		{
			LPSTR messageBuffer = nullptr;
			size_t size = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
			applog(LOG_ERR, "Error while trying to set the priority:");
			applog(LOG_ERR, "%s", messageBuffer);
			LocalFree(messageBuffer);
		}
		prio = GetPriorityClass(GetCurrentProcess());
		switch (prio)
		{
			case NORMAL_PRIORITY_CLASS:
				applog(LOG_INFO, "CPU priority: %s", "normal");
				break;
			case BELOW_NORMAL_PRIORITY_CLASS:
				applog(LOG_INFO, "CPU priority: %s", "below normal");
				break;
			case ABOVE_NORMAL_PRIORITY_CLASS:
				applog(LOG_INFO, "CPU priority: %s", "above normal");
				break;
			case HIGH_PRIORITY_CLASS:
				applog(LOG_INFO, "CPU priority: %s", "high");
				break;
			case REALTIME_PRIORITY_CLASS:
				applog(LOG_INFO, "CPU priority: %s", "realtime");
				break;
			case IDLE_PRIORITY_CLASS:
				applog(LOG_INFO, "CPU priority: %s", "idle");
				break;
			default:
				applog(LOG_INFO, "CPU priority class: %d", prio);
		}
	}
#endif
	if (opt_affinity != -1) {
		if (!opt_quiet)
			applog(LOG_DEBUG, "Binding process to cpu mask %x", opt_affinity);
		affine_to_cpu_mask(-1, opt_affinity);
	}
	if (active_gpus == 0) {
		applog(LOG_ERR, "No CUDA devices found! terminating.");
		exit(1);
	}
	if (!opt_n_threads)
		opt_n_threads = active_gpus;



// set memspeed /clockspeed
	if (device_memspeed[0] != 0 || device_gpuspeed[0] != 0)
	{
#ifdef WIN32
		applog(LOG_ERR, "Trying to set coreclock to: %d, and memclock to: %d", device_gpuspeed[0],device_memspeed[0]);

		char path[1024];
		system("C:\\Progra~1\\NVIDIA~1\\NVSMI\\nvidia-smi -acp 0");
		for (int i = 0; i < active_gpus; i++)
		{
			sprintf(path, "C:\\Progra~1\\NVIDIA~1\\NVSMI\\nvidia-smi -i %d -ac %d,%d", device_map[i], device_memspeed[0], device_gpuspeed[0]);
			system(path);
		}
#else
		applog(LOG_ERR, "Change the clock is not supported on linux");

#endif
	}


#ifdef HAVE_SYSLOG_H
	if (use_syslog)
		openlog(opt_syslog_pfx, LOG_PID, LOG_USER);
#endif

	work_restart = (struct work_restart *)calloc(opt_n_threads, sizeof(*work_restart));
	if (!work_restart)
		return 1;

	thr_info = (struct thr_info *)calloc(opt_n_threads + 4, sizeof(*thr));
	if (!thr_info)
		return 1;

	/* init workio thread info */
	work_thr_id = opt_n_threads;
	thr = &thr_info[work_thr_id];
	thr->id = work_thr_id;
	thr->q = tq_new();
	if (!thr->q)
		return 1;

	/* start work I/O thread */
	if (pthread_create(&thr->pth, NULL, workio_thread, thr)) {
		applog(LOG_ERR, "workio thread create failed");
		return 1;
	}

	if (want_longpoll && !have_stratum) {
		/* init longpoll thread info */
		longpoll_thr_id = opt_n_threads + 1;
		thr = &thr_info[longpoll_thr_id];
		thr->id = longpoll_thr_id;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		/* start longpoll thread */
		if (unlikely(pthread_create(&thr->pth, NULL, longpoll_thread, thr))) {
			applog(LOG_ERR, "longpoll thread create failed");
			return 1;
		}
	}

	if (want_stratum) {
		/* init stratum thread info */
		stratum_thr_id = opt_n_threads + 2;
		thr = &thr_info[stratum_thr_id];
		thr->id = stratum_thr_id;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		/* start stratum thread */
		if (unlikely(pthread_create(&thr->pth, NULL, stratum_thread, thr))) {
			applog(LOG_ERR, "stratum thread create failed");
			return 1;
		}

		if (have_stratum)
			tq_push(thr_info[stratum_thr_id].q, strdup(rpc_url));
	}

#ifdef USE_WRAPNVML
#ifndef WIN32
	/* nvml is currently not the best choice on Windows (only in x64) */
	hnvml = nvml_create();
	if (hnvml)
		applog(LOG_INFO, "NVML GPU monitoring enabled.");
#else
	if (nvapi_init() == 0)
		applog(LOG_INFO, "NVAPI GPU monitoring enabled.");
#endif
	else
		applog(LOG_INFO, "GPU monitoring is not available.");
#endif

	if (opt_api_listen) {
		/* api thread */
		api_thr_id = opt_n_threads + 3;
		thr = &thr_info[api_thr_id];
		thr->id = api_thr_id;
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		/* start stratum thread */
		if (unlikely(pthread_create(&thr->pth, NULL, api_thread, thr))) {
			applog(LOG_ERR, "api thread create failed");
			return 1;
		}
	}

	/* start mining threads */
	for (i = 0; i < opt_n_threads; i++) {
		thr = &thr_info[i];

		thr->id = i;
		thr->gpu.thr_id = i;
		thr->gpu.gpu_id = (uint8_t) device_map[i];
		thr->gpu.gpu_arch = (uint16_t) device_sm[device_map[i]];
		thr->q = tq_new();
		if (!thr->q)
			return 1;

		if (unlikely(pthread_create(&thr->pth, NULL, miner_thread, thr))) {
			applog(LOG_ERR, "thread %d create failed", i);
			return 1;
		}
	}

	applog(LOG_INFO, "%d miner thread%s started, "
		"using '%s' algorithm.",
		opt_n_threads, opt_n_threads > 1 ? "s":"",
		algo_names[opt_algo]);

#ifdef WIN32
	timeBeginPeriod(1); // enable high timer precision (similar to Google Chrome Trick)
#endif

	/* main loop - simply wait for workio thread to exit */
	pthread_join(thr_info[work_thr_id].pth, NULL);

	/* wait for mining threads */
	for (i = 0; i < opt_n_threads; i++)
		pthread_join(thr_info[i].pth, NULL);
#ifdef WIN32
	timeEndPeriod(1); // be nice and forego high timer precision
#endif
	if (opt_debug)
		applog(LOG_INFO, "workio thread dead, exiting.");

	proper_exit(0);

	return 0;
}
