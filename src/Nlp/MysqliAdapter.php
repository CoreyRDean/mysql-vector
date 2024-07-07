<?php

namespace MHz\MysqlVector\Nlp;

use MHz\MysqlVector\Nlp\DatabaseAdapterInterface;
use mysqli_result;

class MysqliAdapter implements DatabaseAdapterInterface
{
    private \mysqli $connection;
    private bool|mysqli_result $result;
    private ?string $lastError = null;
    private ?string $sql = null;
    private ?\mysqli_stmt $stmt = null;
    public function __construct(\mysqli $mysqli)
    {
        $this->connection = $mysqli;
    }

    public function connect(string $host, string $user, string $password, string $database): MysqliAdapter
    {
        $this->disconnect();
        $this->connection = new \mysqli($host, $user, $password, $database);
        $this->collectError();

        return $this;
    }

    public function query(string $sql): MysqliAdapter
    {
        $this->result = $this->connection->query($sql);
        $this->collectError();

        return $this;
    }

    public function close(): MysqliAdapter
    {
        $this->stmt->close();

        return $this;
    }

    public function begin_transaction(): DatabaseAdapterInterface
    {
        $this->connection->begin_transaction();

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
        $this->connection->rollback();

        return $this;
    }

    public function commit(): DatabaseAdapterInterface
    {
        $this->connection->commit();

        return $this;
    }

    public function prepare(string $sql): DatabaseAdapterInterface
    {
        $this->sql = $sql;
        $this->stmt = $this->connection->prepare($sql);

        return $this;
    }

    public function execute(...$values): DatabaseAdapterInterface
    {
        if (!empty($values)) {
            if (!$this->stmt->bind_param(array_shift($values), ...$values)) {
                $this->collectError();
                return $this;
            }
        }

        if (!$this->stmt->execute()) {
            $this->collectError();
            return $this;
        }

        $this->result = $this->stmt->get_result();
        $this->collectError();
        return $this;
    }

    public function fetch(): ?array
    {
        return $this->result ? $this->result->fetch_assoc() : null;
    }

    public function lastInsertId(): int|string
    {
        return $this->connection->insert_id;
    }

    public function disconnect(): MysqliAdapter
    {
        $this->connection->close();

        return $this;
    }

    private function collectError(): void
    {
        $this->lastError = null;
        if (!empty($this->connection->error)) {
            $this->lastError = $this->connection->error;
        }
    }


}