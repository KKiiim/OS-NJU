#include <stdio.h>
#include <dirent.h>
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include <stdlib.h>

struct ProcMessage_s
{
    const char* name;
    int Pid;
    int PPid;
} ProcMessage_t;

void getProcMessage(const char* name, struct ProcMessage_s* pMessage)
{
    char* procPath = "/proc/";
    char* st = "/status";
    // printf("name %s\n", name);
    char* path = (char*)malloc(strlen(procPath) + strlen(name) + strlen(st));
    strcpy(path, procPath);
    strcat(path, name);
    strcat(path, st);
    // printf("path %s\n", path);

    assert(path != NULL);

    // Segmentation fault (core dumped)
    // DIR* procDir = opendir(path);
    // if (procDir == NULL)
    // {
    //     printf("cant reach proc %s\n", procPath);
    //     return;
    // }

    FILE* status = fopen(path, "r");
    if (status == NULL)
    {
        printf("cant reach proc %s\n status", name);
        return;
    }
    // printf("get proc %s\n", name);
    char* thisLine = NULL;
    size_t len = 0;
    // int cout = 1;
    // while (getline(&thisLine, &len, status) != -1)
    // {
    //     if (cout-- == 0)
    //         break;
    //     printf("%s", thisLine);
    // }

    getline(&thisLine, &len, status);
    // NAME
    printf("%s", thisLine);
    getline(&thisLine, &len, status);
    getline(&thisLine, &len, status);
    getline(&thisLine, &len, status);
    getline(&thisLine, &len, status);
    getline(&thisLine, &len, status);
    // Pid
    printf("%s", thisLine);
    getline(&thisLine, &len, status);
    // PPid
    printf("%s\n\n", thisLine);

}

bool inline isProc(const char* name)
{
    for (int index = 0; name[index] != '\0'; ++index)
    {
        if (name[index] < '0' || name[index] > '9')
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    for (int i = 0; i < argc; i++)
    {
        assert(argv[i]);
        // printf("argv[%d] = %s\n", i, argv[i]);
    }
    assert(!argv[argc]);

    FILE* proc = fopen("/proc", "r");
    DIR* rootDir = opendir("/proc");
    assert(rootDir != NULL);

    struct dirent* dir;
    while ((dir = readdir(rootDir)) != NULL)
    {
        assert(dir->d_name != NULL);
        if (!isProc(dir->d_name))
        {
            continue;
        }
        // printf("%s\n", dir->d_name);
        struct ProcMessage_s pMessage = {
            NULL,
            -1,
            -1,
        };
        getProcMessage(dir->d_name, &pMessage);
        //assert(pMessage.name != NULL && pMessage.Pid != -1 && pMessage.PPid != -1);
    }

    fclose(proc);

    return 0;
}
