/* PolarQuant KV tier (kv_tq.h). The rotation must be orthonormal and self-inverse,
 * the polar transform must preserve the radius EXACTLY (only the direction is lossy),
 * the angle grids must round-trip their own centroids, and the packed row must
 * survive quant->dequant. Distortion is data-oblivious: because the randomized
 * Hadamard rotation gaussianizes ANY input, even a one-hot vector reconstructs well.
 * Everything is deterministic (fixed seed), so no calibration and no fixtures. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../kv_tq.h"

static int fails=0;
#define CHECK(cond, ...) do{ if(!(cond)){ fails++; \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); fprintf(stderr,__VA_ARGS__); fputc('\n',stderr);} }while(0)

/* deterministic pseudo-random floats (no libc rand: reproducible across platforms) */
static uint32_t rng=0x12345678u;
static float frand(void){ rng=rng*1664525u+1013904223u; return (float)((rng>>8)&0xFFFFFF)/(float)0x1000000 * 2.f - 1.f; }
static float gauss(void){ /* Box-Muller-ish from two uniforms */
    float u=(frand()+1.f)*0.5f+1e-7f, v=(frand()+1.f)*0.5f;
    return sqrtf(-2.f*logf(u))*cosf(2.f*COLI_TQ_PI*v);
}

static float norm2(const float *a,int n){ double s=0; for(int i=0;i<n;i++) s+=(double)a[i]*a[i]; return (float)sqrt(s); }
static float l2diff(const float *a,const float *b,int n){ double s=0; for(int i=0;i<n;i++){ double d=(double)a[i]-b[i]; s+=d*d; } return (float)sqrt(s); }

/* average relative L2 reconstruction error over many random Gaussian rows */
static float avg_relerr(int n,int bits,int rows){
    int rb=coli_tq_row_bytes(n,bits);
    float *x=malloc(sizeof(float)*n), *y=malloc(sizeof(float)*n); uint8_t *q=malloc(rb);
    double acc=0;
    for(int r=0;r<rows;r++){
        for(int i=0;i<n;i++) x[i]=gauss()*3.7f;
        float rad=coli_tq_quant_row(x,q,n,bits);
        coli_tq_dequant_row(q,rad,y,n,bits);
        float nx=norm2(x,n); acc += (nx>0)? l2diff(x,y,n)/nx : 0;
    }
    free(x); free(y); free(q);
    return (float)(acc/rows);
}

int main(void){
    /* ---- FWHT is normalized and self-inverse ---- */
    for(int n=2; n<=1024; n<<=1){
        float a[1024], b[1024];
        for(int i=0;i<n;i++){ a[i]=gauss(); b[i]=a[i]; }
        coli_tq_fwht(b,n); coli_tq_fwht(b,n);
        float e=l2diff(a,b,n), na=norm2(a,n);
        CHECK(e <= 1e-4f*(na+1.f), "FWHT self-inverse n=%d: err %g", n, e);
        /* one application preserves the L2 norm (orthonormal) */
        for(int i=0;i<n;i++) b[i]=a[i];
        coli_tq_fwht(b,n);
        CHECK(fabsf(norm2(b,n)-na) <= 1e-4f*(na+1.f), "FWHT preserves norm n=%d", n);
    }

    /* ---- rotate/unrotate is the identity and preserves the norm ---- */
    for(int n=2; n<=1024; n<<=1){
        float x[1024], y[1024], z[1024];
        for(int i=0;i<n;i++) x[i]=gauss()*2.5f;
        coli_tq_rotate(x,y,n,COLI_TQ_SEED);
        CHECK(fabsf(norm2(y,n)-norm2(x,n)) <= 1e-4f*(norm2(x,n)+1.f), "rotate preserves norm n=%d", n);
        memcpy(z,y,sizeof(float)*n);
        coli_tq_unrotate(z,n,COLI_TQ_SEED);
        CHECK(l2diff(x,z,n) <= 1e-4f*(norm2(x,n)+1.f), "rotate->unrotate identity n=%d", n);
    }

    /* ---- angle grids round-trip their own codes (centroid re-encodes to itself) ---- */
    for(int b=1;b<=6;b++){
        for(uint32_t c=0;c<(1u<<b);c++){
            float ctr=coli_tq_dec1(c,b);
            CHECK(coli_tq_enc1(ctr,b)==c, "level-1 grid b=%d code %u", b, c);
        }
        for(int lvl=2; lvl<=9; lvl++){
            for(uint32_t c=0;c<(1u<<b);c++){
                float ctr=coli_tq_dec2(c,b,lvl);
                CHECK(coli_tq_enc2(ctr,b,lvl)==c, "centered grid b=%d lvl=%d code %u", b, lvl, c);
            }
        }
    }

    /* ---- bit-cursor packs and unpacks any width losslessly ---- */
    {
        uint8_t buf[64]; memset(buf,0,sizeof(buf));
        int wpos=0; uint32_t vals[20];
        for(int i=0;i<20;i++){ int w=1+(i%6); vals[i]=coli_tq_hash((uint32_t)i)&((1u<<w)-1); coli_tq_wbits(buf,&wpos,vals[i],w); }
        int rpos=0;
        for(int i=0;i<20;i++){ int w=1+(i%6); uint32_t v=coli_tq_rbits(buf,&rpos,w); CHECK(v==vals[i], "bitcursor i=%d w=%d %u!=%u", i, w, v, vals[i]); }
        CHECK(wpos==rpos, "bitcursor cursor mismatch %d %d", wpos, rpos);
    }

    /* ---- row_bytes matches the hand count: (n/2)*b1 + (n/2-1)*b2 bits ---- */
    CHECK(coli_tq_row_bytes(512,4)==(256*4+255*3+7)/8, "row_bytes 512/4 = %d", coli_tq_row_bytes(512,4));
    CHECK(coli_tq_row_bytes(512,3)==(256*3+255*2+7)/8, "row_bytes 512/3 = %d", coli_tq_row_bytes(512,3));
    CHECK(coli_tq_row_bytes(64,4) ==( 32*4+ 31*3+7)/8, "row_bytes 64/4 = %d",  coli_tq_row_bytes(64,4));
    /* it must be smaller than the KV8 byte count (n bytes/row) — that's the point */
    CHECK(coli_tq_row_bytes(512,4) < 512, "TQ4 row < KV8 row (512): %d", coli_tq_row_bytes(512,4));

    /* ---- full-row round-trip: radius EXACT, direction lossy but bounded ---- */
    for(int n=64; n<=512; n<<=1){
        for(int bits=3; bits<=4; bits++){
            int rb=coli_tq_row_bytes(n,bits);
            float x[512], y[512]; uint8_t q[512];
            for(int i=0;i<n;i++) x[i]=gauss()*3.7f;
            float rad=coli_tq_quant_row(x,q,n,bits);
            CHECK(fabsf(rad-norm2(x,n))<=1e-4f*norm2(x,n), "radius==||x|| n=%d bits=%d (%g vs %g)", n, bits, rad, norm2(x,n));
            coli_tq_dequant_row(q,rad,y,n,bits);
            CHECK(fabsf(norm2(y,n)-rad)<=1e-3f*rad, "recon preserves radius n=%d bits=%d (%g vs %g)", n, bits, norm2(y,n), rad);
            (void)rb;
        }
    }

    /* ---- full-pipeline DIRECTION coverage at every production size ----
     * norm preservation alone is a tautology (decode is radius-preserving by
     * construction), so a wrong seed / wrong angle-offset in decode would corrupt
     * direction while ||y|| stayed exactly right. avg_relerr compares x vs y
     * element-wise, so it — and only it — catches that. n=64 is the real qk_rope
     * width, n=512 the kv_lora latent; both must have direction coverage. */
    for(int n=64; n<=512; n<<=1){
        float e3=avg_relerr(n,3,150), e4=avg_relerr(n,4,150);
        CHECK(e4 < e3, "n=%d: more bits -> less distortion (e4=%g e3=%g)", n, e4, e3);
        CHECK(e4 < 0.45f, "n=%d: TQ4 mean rel err small (%g)", n, e4);
        CHECK(e3 < 0.65f, "n=%d: TQ3 mean rel err bounded (%g)", n, e3);
        fprintf(stderr,"[info] mean rel L2 err n=%d: TQ3=%.4f TQ4=%.4f\n", n, e3, e4);
    }

    /* ---- preconditioning handles adversarial (non-Gaussian) input ----
     * a one-hot vector rotates to Hadamard-column form (all coords +-1/sqrt(n)), so
     * its deeper sub-radii are all equal and those angles land exactly on pi/4 (a
     * grid center). The level-1 angles sit at +-pi/4, midway between the uniform
     * bins, so error stays at the same grid floor as Gaussian input (not worse) —
     * the point is a maximally-spread input still round-trips with bounded error and
     * an EXACT radius, never a NaN. */
    {
        int n=256, bits=4, rb=coli_tq_row_bytes(n,bits);
        float x[256]={0}, y[256]; uint8_t *q=malloc(rb);
        x[97]=1.f;
        float rad=coli_tq_quant_row(x,q,n,bits);
        coli_tq_dequant_row(q,rad,y,n,bits);
        CHECK(fabsf(rad-1.f)<1e-5f, "one-hot radius 1 (%g)", rad);
        CHECK(l2diff(x,y,n) < 0.35f, "one-hot round-trips within the grid floor (%g)", l2diff(x,y,n));
        free(q);
    }

    /* ---- inert rows: zero / NaN / Inf / subnormal -> radius 0, decode all-zero ---- */
    {
        int n=64, bits=4, rb=coli_tq_row_bytes(n,bits);
        float x[64], y[64]; uint8_t *q=malloc(rb);
        /* zero */
        memset(x,0,sizeof(float)*n);
        CHECK(coli_tq_quant_row(x,q,n,bits)==0.f, "zero row radius 0");
        coli_tq_dequant_row(q,0.f,y,n,bits);
        for(int i=0;i<n;i++) CHECK(y[i]==0.f, "zero row decode [%d]", i);
        /* NaN poisons the norm -> inert */
        for(int i=0;i<n;i++) x[i]=gauss(); x[7]=NAN;
        CHECK(coli_tq_quant_row(x,q,n,bits)==0.f, "NaN row -> inert radius 0");
        /* Inf -> inert */
        for(int i=0;i<n;i++) x[i]=gauss(); x[7]=INFINITY;
        CHECK(coli_tq_quant_row(x,q,n,bits)==0.f, "Inf row -> inert radius 0");
        /* subnormal-magnitude row: norm below the guard -> inert (like KV8) */
        for(int i=0;i<n;i++) x[i]=1e-38f;
        CHECK(coli_tq_quant_row(x,q,n,bits)==0.f, "subnormal row -> inert radius 0");
        /* a non-finite radius read from a corrupt file decodes to zero, never NaN */
        for(int i=0;i<n;i++) x[i]=gauss();
        float rad=coli_tq_quant_row(x,q,n,bits);
        coli_tq_dequant_row(q,NAN,y,n,bits);
        for(int i=0;i<n;i++) CHECK(y[i]==0.f, "NaN radius decodes to zero [%d]", i);
        (void)rad;
        free(q);
    }

    /* ---- rotated int4 codec (coli_q4): radius exact, LOWER distortion than polar at 4-bit,
     * full reconstruction, inert rows. This is the competitive 4-bit tier. ---- */
    for(int n=64; n<=512; n<<=1){
        CHECK(coli_q4_row_bytes(n)==n/2, "q4 row bytes n/2 (n=%d)", n);
        double sq=0, sp=0; int rows=200;
        for(int r=0;r<rows;r++){
            float x[512], y[512], yp[512]; uint8_t q[256], qp[512];
            for(int i=0;i<n;i++) x[i]=gauss()*3.7f;
            float rad=coli_q4_quant_row(x,q,n); coli_q4_dequant_row(q,rad,y,n);
            float radp=coli_tq_quant_row(x,qp,n,4); coli_tq_dequant_row(qp,radp,yp,n,4);
            /* radius stored = ||x|| exactly; unlike polar, int4 recon does NOT preserve the norm
             * (each coord is quantized), so only check the stored radius here. */
            if(r==0) CHECK(fabsf(rad-norm2(x,n))<=1e-4f*norm2(x,n), "q4 radius==||x|| n=%d", n);
            sq += l2diff(x,y,n)/norm2(x,n); sp += l2diff(x,yp,n)/norm2(x,n);
        }
        sq/=rows; sp/=rows;
        CHECK(sq < sp, "n=%d: rotated-int4 beats PolarQuant at 4-bit (%.4f vs %.4f)", n, sq, sp);
        fprintf(stderr,"[info] rel L2 err n=%d: int4=%.4f polar4=%.4f\n", n, sq, sp);
    }
    { int n=128; float x[128], y[128]; uint8_t q[64];
      memset(x,0,sizeof(float)*n); CHECK(coli_q4_quant_row(x,q,n)==0.f, "q4 zero row inert");
      coli_q4_dequant_row(q,0.f,y,n); for(int i=0;i<n;i++) CHECK(y[i]==0.f, "q4 zero decode [%d]", i);
      for(int i=0;i<n;i++) x[i]=gauss(); x[3]=NAN; CHECK(coli_q4_quant_row(x,q,n)==0.f, "q4 NaN row inert");
      for(int i=0;i<n;i++) x[i]=gauss(); x[3]=INFINITY; CHECK(coli_q4_quant_row(x,q,n)==0.f, "q4 Inf row inert"); }

    if(fails){ fprintf(stderr,"%d failure(s)\n",fails); return 1; }
    printf("OK kv_tq (PolarQuant + rotated-int4 codec: rotation, roundtrip, distortion, inert rows)\n");
    return 0;
}
