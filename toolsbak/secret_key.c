// 简化的密钥存储
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

// 直接存储完整密钥
static const char secret_key[] = "nfA*vz1jtyvJ3ca5BV=1yN@o-z9HCY~,%f=xkfm,W]-09.%--VcDZAK-VTN-m6DZ^AD}ga_YbXm-Qmo#eW%fgCi+#N)oHs9m.PUh";

// 获取SECRET_KEY
EXPORT const char* get_secret_key() {
    return secret_key;
}
