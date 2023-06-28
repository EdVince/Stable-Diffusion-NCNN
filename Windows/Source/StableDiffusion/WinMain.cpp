#include <Windows.h>

#include "pch.h"

#include "App.h"
#include "MainPage.h"

#include "StableDiffusionCore.h"

int WINAPI wWinMain(
    _In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR lpCmdLine,
    _In_ int nShowCmd)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    winrt::init_apartment(winrt::apartment_type::single_threaded);

    winrt::com_ptr<winrt::StableDiffusion::implementation::App> app =
        winrt::make_self<winrt::StableDiffusion::implementation::App>();

    winrt::StableDiffusion::MainPage XamlWindowContent =
        winrt::make<winrt::StableDiffusion::implementation::MainPage>();

    HWND WindowHandle = ::CreateWindowExW(
        WS_EX_CLIENTEDGE,
        L"Mile.Xaml.ContentWindow",
        L"NCNN-based Stable Diffusion GUI",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        0,
        CW_USEDEFAULT,
        0,
        nullptr,
        nullptr,
        hInstance,
        winrt::get_abi(XamlWindowContent));
    if (!WindowHandle)
    {
        return -1;
    }

    StableDiffusion::MainWindowHandle = WindowHandle;

    ::ShowWindow(WindowHandle, nShowCmd);
    ::UpdateWindow(WindowHandle);

    MSG Message;
    while (::GetMessageW(&Message, nullptr, 0, 0))
    {
        // Workaround for capturing Alt+F4 in applications with XAML Islands.
        // Reference: https://github.com/microsoft/microsoft-ui-xaml/issues/2408
        if (Message.message == WM_SYSKEYDOWN && Message.wParam == VK_F4)
        {
            ::SendMessageW(
                ::GetAncestor(Message.hwnd, GA_ROOT),
                Message.message,
                Message.wParam,
                Message.lParam);

            continue;
        }

        ::TranslateMessage(&Message);
        ::DispatchMessageW(&Message);
    }

    app->Close();

    return static_cast<int>(Message.wParam);
}
