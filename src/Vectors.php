<?php

namespace MHz\MysqlVector;

use MHz\MysqlVector\Nlp\DatabaseAdapterInterface;
use MHz\MysqlVector\Nlp\Embedder;

class Vectors
{
    private VectorTable $table;
    private DatabaseAdapterInterface $db;
    private Embedder $embedder;
    private ?VectorTable $currentTable;
    private int $currentVectorDimension;
    private int $getMaxInputLength;

    private string $currentTableName = 'general';
    private string $currentEngine = 'InnoDB';
    private ?string $currentTablePrefix = null;
    private ?string $currentTableSuffix = null;
    public static function isAvailable(): bool
    {
        return extension_loaded('ffi') && in_array(ini_get('ffi.enable'), ['true', '1'], true);
    }

    public static function createWithMysqli(\mysqli $mysqli): Vectors
    {
        return new self(new Nlp\MysqliAdapter($mysqli));
    }

    public static function createWithPdo(\PDO $pdo): Vectors
    {
        return new self(new Nlp\PdoAdapter($pdo));
    }

    public function __construct(DatabaseAdapterInterface $dba)
    {
        $this->embedder = new Embedder();
        $this->currentVectorDimension = $this->embedder->getDimensions();
        $this->getMaxInputLength = $this->embedder->getMaxLength();
        $this->db = $dba;
    }

    public function usingCategory(string $category): Vectors
    {
        $this->currentTableName = $category;
        $this->currentTable = null;

        return $this;
    }

    public function doesCategoryExist(string $category): bool
    {
        return $this->getTable(false)->doesTableExist();
    }

    public function overrideVectorDimension(int $dimension): Vectors
    {
        $this->currentVectorDimension = $dimension;
        $this->currentTable = null;

        return $this;
    }

    public function overrideEngine(string $engine): Vectors
    {
        $this->currentEngine = $engine;
        $this->currentTable = null;

        return $this;
    }

    public function overrideTableFixes(string $tablePrefix, string $tableSuffix = null): Vectors
    {
        $this->currentTablePrefix = $tablePrefix;
        $this->currentTableSuffix = $tableSuffix;
        $this->currentTable = null;

        return $this;
    }

    public function forceRecreateTable(): Vectors
    {
        $this->getTable()->drop();
        $this->currentTable = null;

        $this->getTable();

        return $this;
    }

    public function destroyTable(): Vectors
    {
        $this->currentTable = new VectorTable($this->db, $this->currentTableName, $this->currentVectorDimension, $this->currentEngine);
        $this->currentTable->drop();

        return $this;
    }

    public function store(string $text): int
    {
        $vector = $this->embed($text);

        if ($id = $this->idOfString($text)) {
            return $id;
        }

        return $this->getTable()->upsert($vector);
    }

    public function getByIds(array $ids): array
    {
        return $this->getTable()->select($ids);
    }

    public function deleteById(array $id): void
    {
        $this->getTable()->delete($id);
    }

    public function updateById(int $id, string $text): void
    {
        $vector = $this->embed($text);
        $this->getTable()->upsert($vector, $id);
    }

    public function search(string $text, int $limit = 10): array
    {
        $vector = $this->embed($text);
        return array_map(function ($vector) {
            return [
                'id' => $vector['id'],
                'similarity' => $vector['similarity']
            ];
        }, $this->getTable()->search($vector, $limit));
    }

    public function searchById(int $id, int $limit = 10): array
    {
        $vectors = $this->getByIds([$id]);

        if (empty($vectors)) {
            throw new \Exception("Vector with id $id not found.");
        }

        $vector = $vectors[0]['vector'];

        return array_map(function ($vector) {
            return [
                'id' => $vector['id'],
                'similarity' => $vector['similarity']
            ];
        }, $this->getTable()->search($vector, $limit));
    }

    public function idOfString(string $text): ?int
    {
        $vector = $this->embed($text);
        try {
            $similar = $this->getTable()->search($vector, 1);
        } catch (\Exception $e) {
            return null;
        }
        if (empty($similar)) {
            return null;
        }
        if ($similar[0]['similarity'] < 0.999) {
            return null;
        }
        return $similar[0]['id'];
    }

    public function similarityToString(string $text1, string $text2): float
    {
        $vector1 = $this->embed($text1, true);
        $vector2 = $this->embed($text2, true);
        return $this->getTable()->cosim($vector1, $vector2);
    }

    public function similarityToId(string $text, int $id): float
    {
        $vector1 = $this->embed($text, true);
        $selectedVectors = $this->getTable()->select([$id]);

        if (empty($selectedVectors)) {
            throw new \Exception("Vector with id $id not found.");
        }

        $vector2 = $selectedVectors[0]['normalized_vector'];
        return $this->getTable()->cosim($vector1, $vector2);
    }

    private function getTable(bool $initialize = true): VectorTable
    {
        $table = $this->currentTable;
        if ($table === null) {
            $table = new VectorTable($this->db, $this->currentTableName, $this->currentVectorDimension, $this->currentEngine);
        }

        if ($this->currentTablePrefix || $this->currentTableSuffix) {
            $table->setTableFixes($this->currentTablePrefix, $this->currentTableSuffix ?? '');
        }

        if ($initialize && !$this->currentTable) {
            $table->initialize();
        }

        $this->currentTable = $table;

        return $this->currentTable;
    }

    private function embed(string $text, bool $normalized = false): array {
        $texts = explode(' ', $text);

        // Split each text entry if it is longer than the maximum length and flatten the array
        $texts = array_merge(...array_map(function($text) {
            return str_split($text, $this->getMaxInputLength);
        }, $texts));

        $embeddings = $this->embedder->embed($texts);
        $vectors = [];
        foreach ($embeddings as $i => $embedding) {
            $vector = $this->calculateMeanVector($embedding);
            $vectors[] = $vector;
        }

        $finalVector = $this->calculateMeanVector($vectors);

        if ($normalized) {
            return $this->normalize($finalVector);
        }

        return $finalVector;
    }

    private function calculateMeanVector(array $embeddings): array {
        $meanVector = [];
        $vectorCount = count($embeddings);

        // Initialize meanVector with zeros based on the dimension of the first embedding
        if ($vectorCount > 0) {
            $meanVector = array_fill(0, $this->currentVectorDimension, 0.0);

            // Sum up all vectors
            foreach ($embeddings as $embedding) {
                foreach ($embedding as $dimension => $value) {
                    $meanVector[$dimension] += $value;
                }
            }

            // Divide by the number of vectors to get the mean
            foreach ($meanVector as &$value) {
                $value /= $vectorCount;
            }
        }

        return $meanVector;
    }

    private function normalize(array $vector): array {
        $magnitude = $this->getTable()->getMagnitude($vector);
        return $this->getTable()->normalize($vector, $magnitude);
    }

}