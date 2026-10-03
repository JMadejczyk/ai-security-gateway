-- Small deterministic seed. Runs as the postgres superuser, which is not subject to RLS.
--
-- Customers per region: north 20, south 13, east 7, west 10 (50 total).
-- Visible customer counts:
--   anna@demo        north + south + east  -> 40
--   bartek@demo      east                  ->  7
--   root@demo        all regions           -> 50
--   svc:nightly_etl  all regions           -> 50
--   olga@demo        none                  ->  0  (ops-team approver, grants nothing)

INSERT INTO acl.region_access (user_id, region) VALUES
    ('anna@demo', 'north'), ('anna@demo', 'south'), ('anna@demo', 'east'),
    ('bartek@demo', 'east'),
    ('root@demo', 'north'), ('root@demo', 'south'), ('root@demo', 'east'), ('root@demo', 'west'),
    ('svc:nightly_etl', 'north'), ('svc:nightly_etl', 'south'),
    ('svc:nightly_etl', 'east'), ('svc:nightly_etl', 'west');

INSERT INTO sales.customers (id, name, region, email)
SELECT
    g,
    format('Customer %s', g),
    CASE WHEN g <= 20 THEN 'north' WHEN g <= 33 THEN 'south' WHEN g <= 40 THEN 'east' ELSE 'west' END,
    format('customer%s@example.com', g)
FROM generate_series(1, 50) AS g;

-- Twenty orders per customer (1000), so a visible customer's orders exceed sql_guard's forced
-- LIMIT (500) and customers x orders x payments is a query the planner prices far above
-- max_cost (demo step 4), while COUNT(*) on customers stays trivially cheap (demo step 1).
INSERT INTO sales.orders (id, customer_id, ordered_at, amount)
SELECT
    g,
    (g - 1) / 20 + 1,
    date '2026-01-01' + (g * 7) % 270,
    ((g * 37) % 900 + 100)::numeric(12, 2)
FROM generate_series(1, 1000) AS g;

-- Every order except each fourth one is paid (750 payments).
INSERT INTO sales.payments (id, order_id, paid_at, amount, method)
SELECT
    o.id,
    o.id,
    o.ordered_at + 14,
    o.amount,
    CASE o.id % 3 WHEN 0 THEN 'card' WHEN 1 THEN 'transfer' ELSE 'blik' END
FROM sales.orders AS o
WHERE o.id % 4 <> 0;

-- Fresh statistics, so planner estimates (and sql_guard's EXPLAIN cost threshold) are stable.
ANALYZE acl.region_access, sales.customers, sales.orders, sales.payments;
