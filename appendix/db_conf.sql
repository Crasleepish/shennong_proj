-- 会修改postgresql.auto.conf文件（优先级高于主配置）
ALTER SYSTEM SET enable_seqscan = off;
ALTER SYSTEM SET max_parallel_workers_per_gather = 0;

-- 重载配置
SELECT pg_reload_conf();

SET enable_seqscan = OFF;
SET max_parallel_workers_per_gather = 0;