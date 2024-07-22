<?php

namespace MHz\MysqlVector\Nlp;

// Dynamic approach to locate and require the vendor/autoload.php
$dir = __DIR__;
while (!file_exists($dir . '/vendor/autoload.php')) {
    $dir = dirname($dir);
    if ($dir === '/' || $dir === '.') {
        // autoload.php not found, handle the error or throw an exception
        throw new \Exception('Unable to locate autoload.php. Please run composer install.');
    }
}

require_once $dir . '/vendor/autoload.php';

use OnnxRuntime\Exception;
use OnnxRuntime\Model;
use OnnxRuntime\Vendor;

class Embedder
{
    private Model $model;
    private string $modelHash;
    private BertTokenizer $tokenizer;

    const QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:";
    const EMBEDDING_DIMENSIONS = 384;
    const MAX_LENGTH = 512;

    /**
     * @throws \Exception
     */
    public function __construct() {
        if (!extension_loaded('ffi')) {
            throw new \Exception('The FFI extension for PHP is required to perform embeddings.');
        }

        // check if onnxruntime is installed
        ob_start();
        Vendor::check();
        ob_end_clean();

        // download model if not exists
        $this->downloadFile(
            'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/resolve/main/onnx/model_quantized.onnx',
            __DIR__ . '/../model/model_quantized.onnx'
        );

        $this->downloadFile(
            'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/resolve/main/tokenizer_config.json',
            __DIR__ . '/../model/tokenizer_config.json'
        );

        $this->downloadFile(
            'https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/resolve/main/tokenizer.json',
            __DIR__ . '/../model/tokenizer.json'
        );

        // load model
        $this->model = new Model(__DIR__ . '/../model/model_quantized.onnx');

        $this->modelHash = md5_file(__DIR__ . '/../model/model_quantized.onnx');

        // load tokenizer configuration
        $tokenizerConfig = json_decode(file_get_contents(__DIR__ . '/../model/tokenizer_config.json'), true);
        $tokenizerJSON = json_decode(file_get_contents(__DIR__ . '/../model/tokenizer.json'), true);

        // load BertTokenizer
        $this->tokenizer = new BertTokenizer($tokenizerJSON, $tokenizerConfig);
    }

    public function getInputs(): array {
        return $this->model->inputs();
    }

    public function getOutputs(): array {
        return $this->model->outputs();
    }

    /**
     * Returns the number of dimensions of the output vector.
     * @return int
     */
    public function getDimensions(): int {
        return $this->model->outputs()[0]['shape'][2];
    }

    /**
     * Calculates the embedding of a text.
     * @param array $text Batch of text to embed
     * @return array Batch of embeddings
     * @throws \Exception
     */
    public function embed(array $text, bool $prependQuery = false): array {

        if($prependQuery) {
            // Add query instruction to text
            $text = array_map(function($t) {
                return self::QUERY_INSTRUCTION . ' ' . $t;
            }, $text);
        }

        $tokens = $this->tokenizer->call($text, [
            'text_pair' => null,
            'add_special_tokens' => true,
            'padding' => true,
            'truncation' => true,
            'max_length' => null,
            'return_tensor' => false
        ]);

        $outputs = $this->model->predict($tokens, outputNames: ['last_hidden_state']);
        return $outputs['last_hidden_state'];
    }

    private function dotProduct(array $a, array $b): float {
        return \array_sum(\array_map(
            function ($a, $b) {
                return $a * $b;
            },
            $a,
            $b
        ));
    }

    private function l2Norm(array $a): float {
        return \sqrt(\array_sum(\array_map(function($x) { return $x * $x; }, $a)));
    }

    private function cosine(array $a, array $b): float {
        $dotproduct = $this->dotProduct($a, $b);
        $normA = $this->l2Norm($a);
        $normB = $this->l2Norm($b);
        return 1.0 - ($dotproduct / ($normA * $normB));
    }

    /**
     * Calculates the cosine similarity between two vectors.
     * @param array $a
     * @param array $b
     * @return float
     */
    public function getCosineSimilarity(array $a, array $b): float {
        return 1.0 - $this->cosine($a, $b);
    }

    public function getMaxLength(): int {
        return $this->tokenizer->modelMaxLength;
    }

    public function getModelHash(): string {
        return $this->modelHash;
    }

    private function downloadFile(string $remote, string $local) {
        $url = $remote;
        $localFilePath = $local;
        $lockFilePath = $localFilePath . '.lock';

        // Step 1: Check if the file already exists
        if (!file_exists($localFilePath)) {
            // Step 2: Check for a lock file indicating another process is downloading the file
            while (file_exists($lockFilePath)) {
                // Check the age of the lock file to avoid waiting indefinitely
                if (time() - filemtime($lockFilePath) > 300) { // 300 seconds = 5 minutes
                    // Lock file is older than 5 minutes, assume the other process failed and remove the lock file
                    unlink($lockFilePath);
                    break;
                }
                // Wait for a bit before checking again
                sleep(5);
            }

            // If the file was downloaded while waiting, return
            if (file_exists($localFilePath)) {
                return;
            }

            // Step 1: Get the directory path from the full file path
            $directoryPath = dirname($localFilePath);

            // Step 2 & 3: Check if the directory exists, if not, create it
            if (!file_exists($directoryPath)) {
                mkdir($directoryPath, 0755, true); // Recursive creation
            }

            // Create a lock file to indicate that download is in progress
            file_put_contents($lockFilePath, "lock");

            $fp = fopen($localFilePath, 'w+');
            if ($fp === false) {
                unlink($lockFilePath);
                throw new Exception("Cannot open file ($localFilePath) for writing.");
            }

            $ch = curl_init($url);
            curl_setopt($ch, CURLOPT_FILE, $fp);
            curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true); // Follow redirects
            curl_setopt($ch, CURLOPT_TIMEOUT, 50);

            // For HTTPS URLs, you might need to disable SSL verification in a non-production environment
            // curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);

            curl_exec($ch);

            if (curl_errno($ch)) {
                fclose($fp);
                unlink($localFilePath); // Remove partially downloaded file
                unlink($lockFilePath);
                throw new Exception(curl_error($ch));
            }

            $statusCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);
            fclose($fp);

            if ($statusCode !== 200) {
                unlink($localFilePath); // Remove partially downloaded file if HTTP status code is not 200
                unlink($lockFilePath);
                throw new Exception("File download failed with HTTP status code: $statusCode");
            }

            // Remove the lock file
            unlink($lockFilePath);
        }
    }
}
