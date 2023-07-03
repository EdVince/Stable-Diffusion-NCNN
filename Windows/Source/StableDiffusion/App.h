#pragma once
#include "App.xaml.g.h"

namespace winrt::StableDiffusion::implementation
{
    class App : public AppT<App>
    {
    public:
        App();
        void Close();
    };
}
