// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.makeup;
import android.annotation.SuppressLint;
import android.content.Context;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends Activity
{
    private ImageView imageView;
    private EditText positivePromptText;
    private EditText negativePromptText;
    private EditText stepText;
    private EditText seedText;

    private Bitmap showBitmap;

    private StableDiffusion sd = new StableDiffusion();
    /** Called when the activity is first created. */
    @SuppressLint("MissingInflatedId")
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        String path = MainActivity.this.getFilesDir().getAbsolutePath();
        String name = "vocab.txt";
        copy(MainActivity.this, name, path, name);
        String file = path+File.separator+name;

        String name1 = "log_sigmas.bin";
        copy(MainActivity.this, name1, path, name1);
        String file1 = path+File.separator+name1;

        boolean ret_init = sd.Init(getAssets(), file, file1);
        if (!ret_init)
        {
            Log.e("MainActivity", "SD Init failed");
        }

        imageView = (ImageView) findViewById(R.id.resView);
        positivePromptText = (EditText) findViewById(R.id.pos);
        negativePromptText = (EditText) findViewById(R.id.neg);
        stepText = (EditText) findViewById(R.id.step);
        seedText = (EditText) findViewById(R.id.seed);

        showBitmap = Bitmap.createBitmap(256,256,Bitmap.Config.ARGB_8888);

        Button buttonInitImage = (Button) findViewById(R.id.init);
        buttonInitImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, 1);
            }
        });

        Button buttonTXT2IMG = (Button) findViewById(R.id.txt2img);
        buttonTXT2IMG.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                new Thread(new Runnable() {
                    public void run() {

                        String postivaePrompt = positivePromptText.getText().toString();
                        String negativePrompt = negativePromptText.getText().toString();
                        int step = Integer.valueOf(stepText.getText().toString());
                        int seed = Integer.valueOf(seedText.getText().toString());

                        sd.txt2imgProcess(showBitmap,step,seed,postivaePrompt,negativePrompt);

                        final Bitmap styledImage = showBitmap.copy(Bitmap.Config.ARGB_8888,true);
                        imageView.post(new Runnable() {
                            public void run() {
                                imageView.setImageBitmap(styledImage);
                                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                            }
                        });
                    }
                }).start();
            }
        });

        Button buttonIMG2IMG = (Button) findViewById(R.id.img2img);
        buttonIMG2IMG.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                new Thread(new Runnable() {
                    public void run() {

                        String postivaePrompt = positivePromptText.getText().toString();
                        String negativePrompt = negativePromptText.getText().toString();
                        int step = Integer.valueOf(stepText.getText().toString());
                        int seed = Integer.valueOf(seedText.getText().toString());

                        sd.img2imgProcess(showBitmap,step,seed,postivaePrompt,negativePrompt);

                        final Bitmap styledImage = showBitmap.copy(Bitmap.Config.ARGB_8888,true);
                        imageView.post(new Runnable() {
                            public void run() {
                                imageView.setImageBitmap(styledImage);
                                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                            }
                        });
                    }
                }).start();
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case 1:
                if (resultCode == RESULT_OK && null != data) {
                    Uri selectedImage = data.getData();
                    try {
                        final Bitmap tmp = decodeUri(selectedImage);
                        showBitmap = Bitmap.createScaledBitmap(tmp,256,256,false);
                        imageView.setImageBitmap(showBitmap);
                    } catch (FileNotFoundException e) {
                        throw new RuntimeException(e);
                    }
                }
                break;
            default:
                break;
        }
    }

    private void copy(Context myContext, String ASSETS_NAME, String savePath, String saveName) {
        String filename = savePath + "/" + saveName;
        File dir = new File(savePath);
        if (!dir.exists())
            dir.mkdir();
        try {
            if (!(new File(filename)).exists()) {
                InputStream is = myContext.getResources().getAssets().open(ASSETS_NAME);
                FileOutputStream fos = new FileOutputStream(filename);
                byte[] buffer = new byte[7168];
                int count = 0;
                while ((count = is.read(buffer)) > 0) {
                    fos.write(buffer, 0, count);
                }
                fos.close();
                is.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 256;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                    || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);

        // Rotate according to EXIF
        int rotate = 0;
        try
        {
            ExifInterface exif = new ExifInterface(getContentResolver().openInputStream(selectedImage));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotate = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotate = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotate = 90;
                    break;
            }
        }
        catch (IOException e)
        {
            Log.e("MainActivity", "ExifInterface IOException");
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotate);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }
}
