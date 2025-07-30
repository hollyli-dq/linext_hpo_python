#include "poset.hpp"

#define FOR_EACH_METHOD_BASE1 \
    FOR_EACH_METHOD_ITEM(exact) \
    FOR_EACH_METHOD_ITEM(armc) \
    FOR_EACH_METHOD_ITEM(relaxtpa) \
    FOR_EACH_METHOD_ITEM(relaxtpa_loose1) \
    FOR_EACH_METHOD_ITEM(relaxtpa_loose2) \
    FOR_EACH_METHOD_ITEM(relaxtpa_avx2) \
    FOR_EACH_METHOD_ITEM(telescope_basic_swap) \
    FOR_EACH_METHOD_ITEM(telescope_basic_gibbs) \
    FOR_EACH_METHOD_ITEM(telescope_decomposition_gibbs)

#ifdef __AVX512F__
#define FOR_EACH_METHOD_BASE2 \
    FOR_EACH_METHOD_BASE1 \
    FOR_EACH_METHOD_ITEM(relaxtpa_avx512_short) \
    FOR_EACH_METHOD_ITEM(relaxtpa_avx512)
#else
#define FOR_EACH_METHOD_BASE2 FOR_EACH_METHOD_BASE1
#endif

#ifdef LINEXT_USE_CUDA
#define FOR_EACH_METHOD \
    FOR_EACH_METHOD_BASE2 \
    FOR_EACH_METHOD_ITEM(relaxtpa_gpu)
#else
#define FOR_EACH_METHOD FOR_EACH_METHOD_BASE2
#endif

#define FOR_EACH_METHOD_ITEM(name) \
    template <int W> \
    void method_ ## name (const Poset<W>&, double, double);
FOR_EACH_METHOD
#undef FOR_EACH_METHOD_ITEM

struct Context {
    string methodName;
    double epsilon;
    double delta;

    template <int W>
    void operator()(const Poset<W>& poset) {
        msg("POSET_READ");
        msg("POSET_BITSET_WORD_COUNT ", W);
        showPosetInfo(poset);

        msg("START_METHOD");
        if(false) {
#define FOR_EACH_METHOD_ITEM(name) \
        } else if(methodName == #name) { \
            method_ ## name (poset, epsilon, delta);
FOR_EACH_METHOD
#undef FOR_EACH_METHOD_ITEM
        } else {
            fail("Unknown method: ", methodName);
        }
    }
};

int main(int argc, char* argv[]) {
    cerr.precision(12);

    if(argc != 5) {
        stringstream usage;
        usage << "Invalid command line\n";
        usage << "Usage: " << argv[0] << " <poset filename> <method> <epsilon> <delta>\n";
        usage << "Available methods:";
#define FOR_EACH_METHOD_ITEM(name) \
        { \
            string m = #name; \
            for(char& c : m) { \
                if(c == '_') { \
                    c = '-'; \
                } \
            } \
            usage << "\n  - " << m; \
        }
FOR_EACH_METHOD
#undef FOR_EACH_METHOD_ITEM
        fail(usage.str());
    }

    Context ctx;

    msg("START");
    msg("HARDWARE_THREAD_COUNT ", HardwareThreadCount);

    string posetFilename = argv[1];
    ctx.methodName = argv[2];
    for(char& c : ctx.methodName) {
        if(c == '-') {
            c = '_';
        }
    }
    ctx.epsilon = fromString<double>(argv[3]);
    ctx.delta = fromString<double>(argv[4]);

    msg("POSET_FILENAME ", posetFilename);
    msg("METHOD_NAME ", ctx.methodName);
    msg("EPSILON ", ctx.epsilon);
    msg("DELTA ", ctx.delta);

    if(!isfinite(ctx.epsilon) || ctx.epsilon <= 0.0) {
        fail("Invalid epsilon");
    }
    if(!isfinite(ctx.delta) || ctx.delta <= 0.0) {
        fail("Invalid delta");
    }

    ifstream fp;
    fp.open(posetFilename);
    if(!fp.good()) {
        fail("Opening poset file failed");
    }
    readPoset(fp, true, ctx);

    return 0;
}
