#include "pch.h"
#include "MainPage.h"
#include "MainPage.g.cpp"

#include <regex>

#include <winrt/Windows.UI.Xaml.Media.Imaging.h>

#include <ShObjIdl_core.h>

#include "StableDiffusionCore.h"

using namespace winrt;
using namespace Windows::UI::Xaml;

namespace winrt::StableDiffusion::implementation
{
    using Windows::System::DispatcherQueuePriority;

    void MainPage::NaturalNumberTextBoxBeforeTextChanging(
        TextBox const& sender,
        TextBoxBeforeTextChangingEventArgs const& args)
    {
        if (args.NewText().empty())
        {
            sender.Text(L"0");
            return;
        }

        if (!std::regex_match(
            args.NewText().c_str(),
            std::wregex(
                L"(|[[:digit:]]+)",
                std::regex_constants::icase)))
        {
            args.Cancel(true);
        }
    }

    void MainPage::IntegerNumberTextBoxBeforeTextChanging(
        TextBox const& sender,
        TextBoxBeforeTextChangingEventArgs const& args)
    {
        if (args.NewText().empty())
        {
            sender.Text(L"0");
            return;
        }

        if (!std::regex_match(
            args.NewText().c_str(),
            std::wregex(
                L"(|-)?(|[[:digit:]]+)",
                std::regex_constants::icase)))
        {
            args.Cancel(true);
        }
    }

    void MainPage::InitializeComponent()
    {
        MainPageT::InitializeComponent();

        this->m_DispatcherQueue = DispatcherQueue::GetForCurrentThread();

        StableDiffusion::WriteLine = [this](
            std::wstring const& Content)
        {
            if (!this->m_DispatcherQueue)
            {
                return;
            }
            this->m_DispatcherQueue.TryEnqueue(
                DispatcherQueuePriority::Normal,
                [=]()
            {
                this->GenerateOutputLoggingTextBox().Text(
                    this->GenerateOutputLoggingTextBox().Text()
                    + Content + L"\r\n");
            });
        };

        this->WidthTextBox().Text(L"256");
        this->HeightTextBox().Text(L"256");

        this->FastModeCheckBox().IsChecked(true);

        this->StepTextBox().Text(L"15");

        this->SeedTextBox().Text(L"42");

        this->PositivePromptTextBox().Text(
            L"floating hair, portrait, ((loli)), ((one girl)), cute face, "
            L"hidden hands, asymmetrical bangs, beautiful detailed eyes, "
            L"eye shadow, hair ornament, ribbons, bowties, buttons, "
            L"pleated skirt, (((masterpiece))), ((best quality)), colorful");

        this->NegativePromptTextBox().Text(
            L"((part of the head)), ((((mutated hands and fingers)))), "
            L"deformed, blurry, bad anatomy, disfigured, poorly drawn face, "
            L"mutation, mutated, extra limb, ugly, poorly drawn hands, "
            L"missing limb, blurry, floating limbs, disconnected limbs, "
            L"malformed hands, blur, out of focus, long neck, long body, "
            L"Octane renderer, lowres, bad anatomy, bad hands, text");
    }

    void MainPage::GenerateButtonClickHandler(
        IInspectable const& sender,
        RoutedEventArgs const& e)
    {
        this->WidthTextBox().IsEnabled(false);
        this->HeightTextBox().IsEnabled(false);
        this->FastModeCheckBox().IsEnabled(false);
        this->StepTextBox().IsEnabled(false);
        this->SeedTextBox().IsEnabled(false);
        this->InputImageFileNameBrowseButton().IsEnabled(false);
        this->PositivePromptTextBox().IsEnabled(false);
        this->NegativePromptTextBox().IsEnabled(false);

        this->SaveImageButton().IsEnabled(false);
        this->GenerateButton().IsEnabled(false);
        this->GenerateOutputLoggingTextBox().Text(L"");

        int Width = std::stoi(this->WidthTextBox().Text().c_str());
        int Height = std::stoi(this->HeightTextBox().Text().c_str());
        bool FastMode = winrt::unbox_value_or(
            this->FastModeCheckBox().IsChecked(),
            false);
        int Step = std::stoi(this->StepTextBox().Text().c_str());
        int Seed = std::stoi(this->SeedTextBox().Text().c_str());
        std::string InputImageFileName = winrt::to_string(
            this->InputImageFileNameTextBox().Text());
        std::string PositivePrompt =
            winrt::to_string(this->PositivePromptTextBox().Text());
        std::string NegativePrompt =
            winrt::to_string(this->NegativePromptTextBox().Text());

        std::thread([=]()
        {
            StableDiffusion::GenerateImage(
                Width,
                Height,
                FastMode,
                Step,
                Seed,
                InputImageFileName,
                PositivePrompt,
                NegativePrompt,
                this->m_GeneratedImageForShow,
                this->m_GeneratedImageForSave);
            if (this->m_GeneratedImageForShow.empty() ||
                this->m_GeneratedImageForSave.empty())
            {
                return;
            }

            if (!this->m_DispatcherQueue)
            {
                return;
            }
            this->m_DispatcherQueue.TryEnqueue(
                DispatcherQueuePriority::Normal,
                [=]()
            {
                using Windows::UI::Xaml::Media::Imaging::WriteableBitmap;

                WriteableBitmap GeneratedImage = WriteableBitmap(Width, Height);
                std::memcpy(
                    GeneratedImage.PixelBuffer().data(),
                    this->m_GeneratedImageForShow.data,
                    Width * Height * 4);
                this->GeneratedImageViewer().Source(GeneratedImage);

                this->WidthTextBox().IsEnabled(true);
                this->HeightTextBox().IsEnabled(true);
                this->FastModeCheckBox().IsEnabled(true);
                this->StepTextBox().IsEnabled(true);
                this->SeedTextBox().IsEnabled(true);
                this->InputImageFileNameBrowseButton().IsEnabled(true);
                this->PositivePromptTextBox().IsEnabled(true);
                this->NegativePromptTextBox().IsEnabled(true);

                this->GenerateButton().IsEnabled(true);
                this->SaveImageButton().IsEnabled(true);       
            });

        }).detach();
    }

    void MainPage::InputImageFileNameBrowseButtonClickHandler(
        IInspectable const& sender,
        RoutedEventArgs const& e)
    {
        std::thread([=]()
        {
            try
            {
                winrt::com_ptr<IFileDialog> FileDialog =
                    winrt::create_instance<IFileDialog>(CLSID_FileOpenDialog);

                DWORD Flags = 0;
                winrt::check_hresult(FileDialog->GetOptions(&Flags));

                Flags |= FOS_FORCEFILESYSTEM;
                Flags |= FOS_NOCHANGEDIR;
                Flags |= FOS_DONTADDTORECENT;
                winrt::check_hresult(FileDialog->SetOptions(Flags));

                static constexpr COMDLG_FILTERSPEC SupportedFileTypes[] =
                {
                    {
                        L"Support Input Image (*.jpg, *.jpeg, *.png, *.tga, "
                        L"*.bmp, *.psd, *.gif, *.hdr, *.pic, *.ppm, *.pgm)",
                        L"*.jpg;*.jpeg;*.png;*.tga;*.bmp;*.psd;*.gif;*.hdr;"
                        L"*.pic;*.ppm;*.pgm"
                    }
                };

                winrt::check_hresult(FileDialog->SetFileTypes(
                    ARRAYSIZE(SupportedFileTypes), SupportedFileTypes));

                // Note: The array is 1-indexed
                winrt::check_hresult(
                    FileDialog->SetFileTypeIndex(1));

                winrt::check_hresult(FileDialog->SetDefaultExtension(
                    L"jpg;jpeg;png;tga;bmp;psd;gif;hdr;pic;ppm;pgm"));

                winrt::check_hresult(
                    FileDialog->Show(StableDiffusion::MainWindowHandle));

                winrt::hstring FilePath;
                {
                    winrt::com_ptr<IShellItem> Result;
                    winrt::check_hresult(FileDialog->GetResult(Result.put()));

                    LPWSTR RawFilePath = nullptr;
                    winrt::check_hresult(Result->GetDisplayName(
                        SIGDN_FILESYSPATH,
                        &RawFilePath));
                    FilePath = winrt::to_hstring(RawFilePath);
                    ::CoTaskMemFree(RawFilePath);
                }

                if (!this->m_DispatcherQueue)
                {
                    return;
                }
                this->m_DispatcherQueue.TryEnqueue(
                    DispatcherQueuePriority::Normal,
                    [=]()
                {
                    this->InputImageFileNameTextBox().Text(FilePath);
                });
            }
            catch (...)
            {

            }

        }).detach();
    }

    void MainPage::SaveImageButtonClickHandler(
        IInspectable const& sender,
        RoutedEventArgs const& e)
    {
        std::thread([=]()
        {
            try
            {
                winrt::com_ptr<IFileDialog> FileDialog =
                    winrt::create_instance<IFileDialog>(CLSID_FileSaveDialog);

                DWORD Flags = 0;
                winrt::check_hresult(FileDialog->GetOptions(&Flags));

                Flags |= FOS_FORCEFILESYSTEM;
                Flags |= FOS_NOCHANGEDIR;
                Flags |= FOS_DONTADDTORECENT;
                winrt::check_hresult(FileDialog->SetOptions(Flags));

                static constexpr COMDLG_FILTERSPEC SupportedFileTypes[] =
                {
                    { L"Generated Image (*.png)", L"*.png" }
                };

                winrt::check_hresult(FileDialog->SetFileTypes(
                    ARRAYSIZE(SupportedFileTypes), SupportedFileTypes));

                // Note: The array is 1-indexed
                winrt::check_hresult(
                    FileDialog->SetFileTypeIndex(1));

                winrt::check_hresult(
                    FileDialog->SetDefaultExtension(L"png"));

                // Default to using the tab title as the file name
                winrt::check_hresult(
                    FileDialog->SetFileName(L"Untitled.png"));

                winrt::check_hresult(
                    FileDialog->Show(StableDiffusion::MainWindowHandle));

                winrt::hstring FilePath;
                {
                    winrt::com_ptr<IShellItem> Result;
                    winrt::check_hresult(FileDialog->GetResult(Result.put()));

                    LPWSTR RawFilePath = nullptr;
                    winrt::check_hresult(Result->GetDisplayName(
                        SIGDN_FILESYSPATH,
                        &RawFilePath));
                    FilePath = winrt::to_hstring(RawFilePath);
                    ::CoTaskMemFree(RawFilePath);
                }

                StableDiffusion::WriteImage(
                    winrt::to_string(FilePath).c_str(),
                    this->m_GeneratedImageForSave);
            }
            catch (...)
            {

            }

        }).detach();
    }
}
