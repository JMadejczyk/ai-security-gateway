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

-- Three orders per customer.
INSERT INTO sales.orders (id, customer_id, ordered_at, amount)
SELECT
    g,
    (g - 1) / 3 + 1,
    date '2026-01-01' + (g * 7) % 270,
    ((g * 37) % 900 + 100)::numeric(12, 2)
FROM generate_series(1, 150) AS g;

-- Every order except each fourth one is paid.
INSERT INTO sales.payments (id, order_id, paid_at, amount, method)
SELECT
    o.id,
    o.id,
    o.ordered_at + 14,
    o.amount,
    CASE o.id % 3 WHEN 0 THEN 'card' WHEN 1 THEN 'transfer' ELSE 'blik' END
FROM sales.orders AS o
WHERE o.id % 4 <> 0;
