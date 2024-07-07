<?php

namespace MHz\MysqlVector\Nlp;

interface DatabaseAdapterInterface {
    public function connect(string $host, string $user, string $password, string $database): DatabaseAdapterInterface;
    public function query(string $sql): DatabaseAdapterInterface;
    public function close(): DatabaseAdapterInterface;
    public function begin_transaction(): DatabaseAdapterInterface;
    public function lastError(): ?string;
    public function rollback(): DatabaseAdapterInterface;
    public function commit(): DatabaseAdapterInterface;
    public function prepare(string $sql): DatabaseAdapterInterface;
    public function execute(...$values): DatabaseAdapterInterface;
    public function fetch(): ?array;
    public function lastInsertId(): int|string;
}