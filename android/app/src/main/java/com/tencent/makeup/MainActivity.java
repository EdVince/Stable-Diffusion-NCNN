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
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity
{
    private ImageView imageView;
    private EditText positivePromptText;
    private EditText negativePromptText;
    private EditText stepText;
    private EditText seedText;

    private Makeup makeup = new Makeup();
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

        boolean ret_init = makeup.Init(getAssets(), file, file1);
        if (!ret_init)
        {
            Log.e("MainActivity", "makeup Init failed");
        }

        imageView = (ImageView) findViewById(R.id.resView);
        positivePromptText = (EditText) findViewById(R.id.pos);
        negativePromptText = (EditText) findViewById(R.id.neg);
        stepText = (EditText) findViewById(R.id.step);
        seedText = (EditText) findViewById(R.id.seed);

        final Bitmap showBitmap = Bitmap.createBitmap(512,512,Bitmap.Config.ARGB_8888);

        Button buttonDetect = (Button) findViewById(R.id.go);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                getWindow().setFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE, WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                new Thread(new Runnable() {
                    public void run() {

                        String postivaePrompt = positivePromptText.getText().toString();
                        String negativePrompt = negativePromptText.getText().toString();
                        int step = Integer.valueOf(stepText.getText().toString());
                        int seed = Integer.valueOf(seedText.getText().toString());

                        makeup.Process(showBitmap,step,seed,postivaePrompt,negativePrompt);
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
}
