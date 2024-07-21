<?php

namespace MHz\MysqlVector;

use KMeans\Space;
use MHz\MysqlVector\Nlp\DatabaseAdapterInterface;

class VectorTable
{
    private string $name;
    private int $dimension;
    private string $engine;
    private array $centroidCache;
    private DatabaseAdapterInterface $mysqli;
    private string $prefix = 'vectors_';
    private string $suffix = '';
    private ?string $currentModelHash = null;

    const SQL_COSIM_FUNCTION = "
CREATE FUNCTION COSIM(v1 JSON, v2 JSON) RETURNS FLOAT DETERMINISTIC BEGIN DECLARE sim FLOAT DEFAULT 0; DECLARE i INT DEFAULT 0; DECLARE len INT DEFAULT JSON_LENGTH(v1); IF JSON_LENGTH(v1) != JSON_LENGTH(v2) THEN RETURN NULL; END IF; WHILE i < len DO SET sim = sim + (JSON_EXTRACT(v1, CONCAT('$[', i, ']')) * JSON_EXTRACT(v2, CONCAT('$[', i, ']'))); SET i = i + 1; END WHILE; RETURN sim; END";
    private int $quantizationSampleSize;

    /**
     * Instantiate a new VectorTable object.
     * @param DatabaseAdapterInterface $mysqli The mysqli connection
     * @param string $name Name of the table.
     * @param int $dimension Dimension of the vectors.
     * @param string $engine The storage engine to use for the tables
     */
    public function __construct(DatabaseAdapterInterface $mysqli, string $name, int $dimension = 384, string $engine = 'InnoDB')
    {
        $this->mysqli = $mysqli;
        $this->name = $name;
        $this->dimension = $dimension;
        $this->engine = $engine;
        $this->centroidCache = [];
    }

    public function getVectorTableName(): string
    {
        return sprintf('%s%s%s', $this->prefix, $this->name, $this->suffix);
    }

    public function setTableFixes(string $prefix, string $suffix = ''): void {
        $this->prefix = $prefix;
        $this->suffix = $suffix;
    }

    public function setCurrentModelHash(string $hash) {
        $this->currentModelHash = $hash;
    }

    public function doesTableExist(): bool {
        $tableName = $this->getVectorTableName();
        $statement = $this->mysqli->prepare("SHOW TABLES LIKE ?");
        $statement->execute('s', $tableName);
        $result = $statement->fetch();
        $statement->close();
        return !empty($result);
    }

    protected function getCreateStatements(bool $ifNotExists = true): array {
        $binaryCodeLengthInBytes = ceil($this->dimension / 8);

        $vectorsQuery =
            "CREATE TABLE %s %s (
                id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                vector JSON,
                normalized_vector JSON,
                magnitude DOUBLE,
                binary_code BINARY(%d),
                model_hash_md5 CHAR(32) DEFAULT NULL,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=%s;";
        $vectorsQuery = sprintf($vectorsQuery, $ifNotExists ? 'IF NOT EXISTS' : '', $this->getVectorTableName(), $binaryCodeLengthInBytes, $this->engine);

        return [$vectorsQuery];
    }

    /**
     * Convert an n-dimensional vector in to an n-bit binary code
     * @param array $vector
     * @return int
     */
    private function vectorToHex(array $vector): string {
        $binary = '';
        foreach ($vector as $value) {
            $binary .= $value > 0 ? '1' : '0';
        }

        // Convert binary string to actual binary data
        $binaryData = '';
        foreach (str_split($binary, 8) as $byte) {
            $binaryData .= chr(bindec($byte));
        }

        // Convert the binary data to hexadecimal
        $hex = bin2hex($binaryData);

        return $hex;
    }

    /**
     * Drops the table
     */
    public function drop(): void {
        $this->mysqli->query("DROP TABLE IF EXISTS " . $this->getVectorTableName());
    }

    /**
     * Create the tables required for storing vectors
     * @param bool $ifNotExists Whether to use IF NOT EXISTS in the CREATE TABLE statements
     * @return void
     * @throws \Exception If the tables could not be created
     */
    public function initialize(bool $ifNotExists = true): void
    {
        $this->mysqli->begin_transaction();
        foreach ($this->getCreateStatements($ifNotExists) as $statement) {
            $success = $this->mysqli->query($statement);
            if ($error = $success->lastError()) {
                $e = new \Exception($error);
                $this->mysqli->rollback();
                throw $e;
            }
        }

        // Add COSIM function
        $this->mysqli->query("DROP FUNCTION IF EXISTS COSIM");
        $res = $this->mysqli->query(self::SQL_COSIM_FUNCTION);

        if($error = $res->lastError()) {
            $e = new \Exception($error);
            $this->mysqli->rollback();
            throw $e;
        }

        $binaryCodeLengthInBytes = ceil($this->dimension / 8);
        $this->mysqli->query("CREATE INDEX idx_binary_code ON " . $this->getVectorTableName() . " (binary_code($binaryCodeLengthInBytes))");
        $this->mysqli->lastError();

        $this->mysqli->commit();
    }

    /**
     * Compute the cosine similarity between two normalized vectors
     * @param array $v1 The first vector
     * @param array $v2 The second vector
     * @return float The cosine similarity between the two vectors [0, 1]
     * @throws \Exception
     */
    public function cosim(array $v1, array $v2): float
    {
        $v1 = json_encode($v1);
        $v2 = json_encode($v2);

        $statement = $this->mysqli->prepare("SELECT COSIM('$v1', '$v2')");

        if($error = $statement->lastError()) {
            $e = new \Exception($error);
            $this->mysqli->rollback();
            throw $e;
        }

        $statement->execute();
        $result = $statement->fetch();
        $statement->close();

        return reset($result);
    }

    /**
     * Insert or update a vector
     * @param array $vector The vector to insert or update
     * @param int|null $id Optional ID of the vector to update
     * @return int The ID of the inserted or updated vector
     * @throws \Exception If the vector could not be inserted or updated
     */
    public function upsert(array $vector, int $id = null): int
    {
        $magnitude = $this->getMagnitude($vector);
        $normalizedVector = $this->normalize($vector, $magnitude);
        $binaryCode = $this->vectorToHex($normalizedVector);
        $tableName = $this->getVectorTableName();

        $modelHash = $this->currentModelHash;

        $insertQuery = empty($id) ?
            "INSERT INTO $tableName (vector, normalized_vector, magnitude, binary_code, model_hash_md5) VALUES (?, ?, ?, UNHEX(?), ?)" :
            "UPDATE $tableName SET vector = ?, normalized_vector = ?, magnitude = ?, binary_code = UNHEX(?), model_hash_md5 = ? WHERE id = $id";

        $statement = $this->mysqli->prepare($insertQuery);
        if($error = $statement->lastError()) {
            $e = new \Exception($error);
            $this->mysqli->rollback();
            throw $e;
        }

        $vector = json_encode($vector);
        $normalizedVector = json_encode($normalizedVector);

        $success = $statement->execute('ssdss', $vector, $normalizedVector, $magnitude, $binaryCode, $modelHash);
        if($error = $statement->lastError()) {
            throw new \Exception($error);
        }

        $id = $statement->lastInsertId();
        $statement->close();

        return $id;
    }

    /**
     * Insert multiple vectors in a single query
     * @param array $vectorArray Array of vectors to insert
     * @return array Array of ids of the inserted vectors
     * @throws \Exception
     */
    public function batchInsert(array $vectorArray): array {
        $tableName = $this->getVectorTableName();
        $modelHash = $this->currentModelHash;

        $statement = $this->getConnection()->prepare("INSERT INTO $tableName (vector, normalized_vector, magnitude, binary_code, model_hash_md5) VALUES (?, ?, ?, UNHEX(?), ?)");
        if($error = $this->getConnection()->lastError()) {
            throw new \Exception("Prepare failed: " . $error);
        }

        $ids = [];
        $this->getConnection()->begin_transaction();
        try {
            foreach ($vectorArray as $vector) {
                $magnitude = $this->getMagnitude($vector);
                $normalizedVector = $this->normalize($vector, $magnitude);
                $binaryCode = $this->vectorToHex($normalizedVector);
                $vectorJson = json_encode($vector);
                $normalizedVectorJson = json_encode($normalizedVector);

                if ($error = $statement->execute('ssdss', $vectorJson, $normalizedVectorJson, $magnitude, $binaryCode, $modelHash)->lastError()) {
                    throw new \Exception("Execute failed: " . $error);
                }

                $ids[] = $statement->lastInsertId();
            }

            $this->getConnection()->commit();
        } catch (\Exception $e) {
            $this->getConnection()->rollback();
            throw $e;
        } finally {
            $statement->close();
        }

        return $ids;
    }

    /**
     * Select one or more vectors by id
     * @param array $ids The ids of the vectors to select
     * @return array Array of vectors
     */
    public function select(array $ids, bool $includeOutdated = false): array {
        $tableName = $this->getVectorTableName();
        $modelHash = $this->currentModelHash;

        $placeholders = implode(', ', array_fill(0, count($ids), '?'));
        $sql = "SELECT id, vector, normalized_vector, magnitude, binary_code FROM $tableName WHERE id IN ($placeholders)";

        if ($modelHash && !$includeOutdated) {
            $sql .= " AND model_hash_md5 = '$modelHash'";
        }

        $statement = $this->mysqli->prepare($sql);
        $types = str_repeat('i', count($ids));

        $refs = [];
        foreach ($ids as $key => $id) {
            $refs[$key] = &$ids[$key];
        }

        $statement->execute(...array_merge([$types], $refs));

        $result = [];
        while ($row = $statement->fetch()) {
            $result[] = [
                'id' => intval($row['id']),
                'vector' => json_decode($row['vector'], true),
                'normalized_vector' => json_decode($row['normalized_vector'], true),
                'magnitude' => $row['magnitude'],
                'binary_code' => $row['binary_code']
            ];
        }

        $statement->close();

        return $result;
    }

    public function selectAll(bool $includeOutdated = false): array {
        $tableName = $this->getVectorTableName();
        $modelHash = $this->currentModelHash;

        $sql = "SELECT id, vector, normalized_vector, magnitude, binary_code FROM $tableName";

        if ($modelHash && !$includeOutdated) {
            $sql .= " WHERE model_hash_md5 = '$modelHash'";
        }

        $statement = $this->mysqli->prepare($sql);

        if ($error = $statement->lastError()) {
            $e = new \Exception($error);
            $this->mysqli->rollback();
            throw $e;
        }

        $statement->execute();

        $result = [];
        while ($row = $statement->fetch()) {
            $result[] = [
                'id' => intval($row['id']),
                'vector' => json_decode($row['vector'], true),
                'normalized_vector' => json_decode($row['normalized_vector'], true),
                'magnitude' => $row['magnitude'],
                'binary_code' => $row['binary_code']
            ];
        }

        $statement->close();

        return $result;
    }


    private function dotProduct(array $vectorA, array $vectorB): float {
        $product = 0;

        foreach ($vectorA as $position => $value) {
            if (isset($vectorB[$position])) {
                $product += $value * $vectorB[$position];
            }
        }

        return $product;
    }

    /**
     * Returns the number of vectors stored in the database
     * @return int The number of vectors
     */
    public function count(): int {
        $tableName = $this->getVectorTableName();
        $statement = $this->mysqli->prepare("SELECT COUNT(id) FROM $tableName");
        $statement->execute();
        $result = $statement->fetch();
        $statement->close();
        return reset($result);
    }

    public function getMagnitude(array $vector): float
    {
        $sum = 0;
        foreach ($vector as $value) {
            $sum += $value * $value;
        }

        return sqrt($sum);
    }

    /**
     * Finds the vectors that are most similar to the given vector
     * @param array $vector The vector to query for
     * @param int $n The number of results to return
     * @return array Array of results containing the id, similarity, and vector
     * @throws \Exception
     */
    public function search(array $vector, int $n = 10, bool $includeOutdated = false): array {
        $tableName = $this->getVectorTableName();
        $normalizedVector = $this->normalize($vector);
        $binaryCode = $this->vectorToHex($normalizedVector);
        $modelHash = $this->currentModelHash;

        // Initial search using binary codes
        $sql = "SELECT id, BIT_COUNT(binary_code ^ UNHEX(?)) AS hamming_distance FROM $tableName";

        if ($modelHash && !$includeOutdated) {
            $sql .= " WHERE model_hash_md5 = '$modelHash'";
        }

        $sql .= " ORDER BY hamming_distance LIMIT $n";

        $statement = $this->mysqli->prepare($sql);

        if($error = $statement->lastError()) {
            $e = new \Exception($error);
            $this->mysqli->rollback();
            throw $e;
        }

        $statement->execute('s', $binaryCode);

        $candidates = [];
        while ($row = $statement->fetch()) {
            $candidates[] = $row['id'];
        }
        $statement->close();

        // Rerank candidates using cosine similarity
        $placeholders = implode(',', array_fill(0, count($candidates), '?'));

        if (empty($placeholders)) {
            return [];
        }

        $normalizedVector = json_encode($normalizedVector);

        $sql = "
        SELECT id, vector, normalized_vector, magnitude, COSIM(normalized_vector, '$normalizedVector') AS similarity
        FROM %s
        WHERE id IN ($placeholders)
        ORDER BY similarity DESC
        LIMIT $n";
        $sql = sprintf($sql, $tableName);

        $statement = $this->mysqli->prepare($sql);

        if($error = $statement->lastError()) {
            $e = new \Exception($error);
            $this->mysqli->rollback();
            throw $e;
        }



        $types = str_repeat('i', count($candidates));

        $statement->execute($types, ...$candidates);

        $results = [];
        while ($row = $statement->fetch()) {
            $results[] = [
                'id' => intval($row['id']),
                'vector' => json_decode($row['vector'], true),
                'normalized_vector' => json_decode($row['normalized_vector'], true),
                'magnitude' => $row['magnitude'],
                'similarity' => floatval($row['similarity'])
            ];
        }

        $statement->close();

        return $results;
    }

    /**
     * Normalize a vector
     * @param array $vector The vector to normalize
     * @param float|null $magnitude The magnitude of the vector. If not provided, it will be calculated.
     * @param float $epsilon The epsilon value to use for normalization
     * @return array The normalized vector
     */
    public function normalize(array $vector, float $magnitude = null, float $epsilon = 1e-10): array {
        $magnitude = !empty($magnitude) ? $magnitude : $this->getMagnitude($vector);
        if ($magnitude == 0) {
            $magnitude = $epsilon;
        }
        foreach ($vector as $key => $value) {
            $vector[$key] = $value / $magnitude;
        }
        return $vector;
    }

    /**
     * Remove a vector from the database
     * @param int $id The id of the vector to remove
     * @return void
     * @throws \Exception
     */
    public function delete(int $id): void {
        $tableName = $this->getVectorTableName();
        $statement = $this->mysqli->prepare("DELETE FROM $tableName WHERE id = ?");
        $success = $statement->execute('i', $id);
        if($error = $success->lastError()) {
            throw new \Exception($error);
        }
        $statement->close();
    }

    public function getConnection(): DatabaseAdapterInterface {
        return $this->mysqli;
    }
}
