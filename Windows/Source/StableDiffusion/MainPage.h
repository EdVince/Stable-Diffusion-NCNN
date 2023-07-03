#pragma once

#include "MainPage.g.h"

#include <winrt/Windows.System.h>

#include <ncnn/mat.h>

namespace winrt::StableDiffusion::implementation
{
    using Windows::Foundation::IInspectable;
    using Windows::System::DispatcherQueue;
    using Windows::UI::Xaml::RoutedEventArgs;
    using Windows::UI::Xaml::Controls::TextBox;
    using Windows::UI::Xaml::Controls::TextBoxBeforeTextChangingEventArgs;

    struct MainPage : MainPageT<MainPage>
    {
        MainPage() = default;

        void NaturalNumberTextBoxBeforeTextChanging(
            TextBox const& sender,
            TextBoxBeforeTextChangingEventArgs const& args);
        void IntegerNumberTextBoxBeforeTextChanging(
            TextBox const& sender,
            TextBoxBeforeTextChangingEventArgs const& args);

        void InitializeComponent();

        void GenerateButtonClickHandler(
            IInspectable const& sender,
            RoutedEventArgs const& e);

        void InputImageFileNameBrowseButtonClickHandler(
            IInspectable const& sender,
            RoutedEventArgs const& e);

        void SaveImageButtonClickHandler(
            IInspectable const& sender,
            RoutedEventArgs const& e);

    private:

        DispatcherQueue m_DispatcherQueue = nullptr;

        ncnn::Mat m_GeneratedImageForShow;
        ncnn::Mat m_GeneratedImageForSave;
    };
}

namespace winrt::StableDiffusion::factory_implementation
{
    struct MainPage : MainPageT<MainPage, implementation::MainPage>
    {
    };
}
