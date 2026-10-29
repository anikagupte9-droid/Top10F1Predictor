package com.anika.f1predictor

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        interpreter = Interpreter(loadModelFile())

        setContent {
            F1PredictorApp(interpreter)
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("f1_top10_int8.tflite")

        val inputStream = fileDescriptor.createInputStream()
        val bytes = inputStream.readBytes()

        return ByteBuffer.allocateDirect(bytes.size)
            .order(ByteOrder.nativeOrder())
            .apply {
                put(bytes)
                rewind()
            }
    }

    override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
}

@Composable
fun F1PredictorApp(interpreter: Interpreter) {

    var fp1 by remember { mutableStateOf("") }
    var fp2 by remember { mutableStateOf("") }
    var fp3 by remember { mutableStateOf("") }
    var qualifying by remember { mutableStateOf("") }

    var probability by remember { mutableStateOf<Float?>(null) }
    var prediction by remember { mutableStateOf("") }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            verticalArrangement = Arrangement.Top,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = "F1 Predictor",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Offline Edge AI • Top-10 Prediction",
                style = MaterialTheme.typography.bodyMedium
            )

            Spacer(modifier = Modifier.height(24.dp))

            PositionInput(
                label = "FP1 Position",
                value = fp1,
                onValueChange = { fp1 = it }
            )

            PositionInput(
                label = "FP2 Position",
                value = fp2,
                onValueChange = { fp2 = it }
            )

            PositionInput(
                label = "FP3 Position",
                value = fp3,
                onValueChange = { fp3 = it }
            )

            PositionInput(
                label = "Qualifying Position",
                value = qualifying,
                onValueChange = { qualifying = it }
            )

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                modifier = Modifier.fillMaxWidth(),
                onClick = {

                    val values = listOf(
                        fp1.toFloatOrNull(),
                        fp2.toFloatOrNull(),
                        fp3.toFloatOrNull(),
                        qualifying.toFloatOrNull()
                    )

                    if (values.all { it != null }) {

                        val raw = values.map { it!! }

                        // StandardScaler parameters from training.
                        val means = floatArrayOf(
                            9.885947046843178f,
                            10.081466395112017f,
                            10.228105906313646f,
                            9.963340122199593f
                        )

                        val scales = floatArrayOf(
                            5.6611034012999815f,
                            5.729251299830547f,
                            5.775228394720603f,
                            5.61300136851328f
                        )

                        // StandardScaler
                        val scaled = FloatArray(4)

                        for (i in 0 until 4) {
                            scaled[i] = (raw[i] - means[i]) / scales[i]
                        }

                        // Full INT8 model input quantization.
                        val inputScale = 0.01327837910503149f
                        val inputZeroPoint = -8

                        val input = Array(1) { ByteArray(4) }

                        for (i in 0 until 4) {
                            var q = kotlin.math.round(
                                scaled[i] / inputScale + inputZeroPoint
                            ).toInt()

                            q = q.coerceIn(-128, 127)

                            input[0][i] = q.toByte()
                        }

                        // INT8 output.
                        val output = Array(1) { ByteArray(1) }

                        interpreter.run(input, output)

                        val outputValue = output[0][0].toInt()

                        val outputScale = 0.00390625f
                        val outputZeroPoint = -128

                        val result =
                            (outputValue - outputZeroPoint) * outputScale

                        probability = result.coerceIn(0f, 1f)

                        prediction =
                            if (probability!! >= 0.5f)
                                "TOP 10"
                            else
                                "OUTSIDE TOP 10"
                    }
                }
            ) {
                Text("Predict")
            }

            Spacer(modifier = Modifier.height(24.dp))

            probability?.let {

                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {

                    Column(
                        modifier = Modifier.padding(20.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {

                        Text(
                            text = "Top-10 Probability",
                            style = MaterialTheme.typography.titleMedium
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = "${"%.2f".format(it * 100)}%",
                            style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = prediction,
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun PositionInput(
    label: String,
    value: String,
    onValueChange: (String) -> Unit
) {
    OutlinedTextField(
        value = value,
        onValueChange = {
            if (it.length <= 3 && it.all { char -> char.isDigit() }) {
                onValueChange(it)
            }
        },
        label = { Text(label) },
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 5.dp),
        singleLine = true,
        keyboardOptions = KeyboardOptions(
            keyboardType = KeyboardType.Number
        )
    )
}