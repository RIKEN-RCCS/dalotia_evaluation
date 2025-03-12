#ifndef _CACHE_FLUSH_H_
#define _CACHE_FLUSH_H_

/** author: Jens Domke */

#ifdef __cplusplus
#  define BEGIN_C_DECLS extern "C" {
#  define END_C_DECLS   }
#else                           /* !__cplusplus */
#  define BEGIN_C_DECLS
#  define END_C_DECLS
#endif                          /* __cplusplus */

BEGIN_C_DECLS

typedef enum levels {
	_CF_L1_ = 1,		/* flushes "only" L1 data cache */
	_CF_L2_ = 2,		/* flushes L1 and L2 data cache */
	_CF_L3_ = 3			/* flushes all caches from L3 down */
} lvl_enum_t;

int cf_init(void);
int cf_flush(lvl_enum_t );
int cf_finalize(void);

END_C_DECLS

#endif                          /* _CACHE_FLUSH_H_ */