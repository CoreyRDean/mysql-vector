<?php

namespace MHz\MysqlVector\Nlp;

use MHz\MysqlVector\Nlp\DatabaseAdapterInterface;

class PdoAdapter implements DatabaseAdapterInterface
{
    private ?\PDO $connection;
    private ?string $lastError = null;
    private ?string $sql = null;
    private ?\PDOStatement $stmt = null;

    public function __construct(\PDO $pdo) {
        $this->connection = $pdo;
    }

    public function connect($host, $user, $password, $database): PdoAdapter
    {
        $this->disconnect();
        $this->connection = new \PDO("mysql:host=$host;dbname=$database", $user, $password);
        $this->collectError();

        return $this;
    }

    public function query($sql): PdoAdapter
    {
        $this->sql = $sql;
        try {
            $this->stmt = $this->connection->query($sql);
            $this->collectError();
        } catch (\PDOException $e) {
            $this->stmt = null;
            $this->lastError = $e->getMessage();
        }

        return $this;
    }

    public function close(): PdoAdapter
    {
        $this->stmt->closeCursor();

        return $this;
    }

    public function begin_transaction(): DatabaseAdapterInterface
    {
        $this->connection->beginTransaction();

        return $this;
    }

    public function lastError(): ?string
    {
        $error = $this->lastError;
        $this->lastError = null;
        return $error;
    }

    public function rollback(): DatabaseAdapterInterface
    {
        if ($this->connection->inTransaction()) {
            $this->connection->rollBack();
        }

        $this->collectError();
        return $this;
    }

    public function commit(): DatabaseAdapterInterface
    {
        if ($this->connection->inTransaction()) {
            $this->connection->commit();
        }

        $this->collectError();
        return $this;
    }

    public function prepare(string $sql): DatabaseAdapterInterface
    {
        $this->sql = $sql;

        if ($stmt = $this->connection->prepare($sql)) {
            $this->stmt = $stmt;
        } else {
            $this->stmt = null;
        }

        $this->collectError();
        return $this;
    }

    public function execute(...$values): DatabaseAdapterInterface
    {
        array_shift($values);
        $this->stmt->execute($values);
        $this->collectError();
        return $this;
    }

    public function fetch(): ?array
    {
        if ($result = $this->stmt->fetch(\PDO::FETCH_ASSOC)) {
            $this->collectError();
            return $result;
        }

        return null;
    }

    public function lastInsertId(): int|string
    {
        return $this->connection->lastInsertId();
    }

    private function collectError(): void
    {
        $this->lastError = null;
        if (!empty($this->connection->errorInfo()[2] ?? null)) {
            $this->lastError = $this->connection->errorInfo()[2] ?? null;
        }
    }

    private function disconnect(): PdoAdapter
    {
        $this->connection = null;

        return $this;
    }
}