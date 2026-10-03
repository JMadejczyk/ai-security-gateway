-- Demo sales database with row-level security.
--
-- * sales_owner owns every object and cannot log in.
-- * acl_app is the role the mcp-postgres server connects as: not the owner, no SUPERUSER,
--   no BYPASSRLS, SELECT only.
-- * Every table has ENABLE + FORCE ROW LEVEL SECURITY, so even the owner is filtered.
-- * Visibility is driven by the transaction-local setting app.user_id, which mcp-postgres sets
--   with set_config('app.user_id', <principal>, true) at the start of each transaction.
--   Unset (or empty) means no rows.

CREATE ROLE sales_owner NOLOGIN;
CREATE ROLE acl_app LOGIN PASSWORD :'app_password'
    NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;

REVOKE ALL ON SCHEMA public FROM PUBLIC;

CREATE SCHEMA sales AUTHORIZATION sales_owner;
-- Access mapping lives outside `sales`, so `read:db:sales.*` grants never cover it.
CREATE SCHEMA acl AUTHORIZATION sales_owner;

SET ROLE sales_owner;

CREATE TABLE acl.region_access (
    user_id text NOT NULL,
    region  text NOT NULL,
    PRIMARY KEY (user_id, region)
);

CREATE TABLE sales.customers (
    id     integer PRIMARY KEY,
    name   text NOT NULL,
    region text NOT NULL,
    email  text NOT NULL
);

CREATE TABLE sales.orders (
    id          integer PRIMARY KEY,
    customer_id integer NOT NULL REFERENCES sales.customers (id),
    ordered_at  date NOT NULL,
    amount      numeric(12, 2) NOT NULL
);

CREATE TABLE sales.payments (
    id       integer PRIMARY KEY,
    order_id integer NOT NULL REFERENCES sales.orders (id),
    paid_at  date NOT NULL,
    amount   numeric(12, 2) NOT NULL,
    method   text NOT NULL
);

ALTER TABLE acl.region_access ENABLE ROW LEVEL SECURITY;
ALTER TABLE acl.region_access FORCE ROW LEVEL SECURITY;
ALTER TABLE sales.customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE sales.customers FORCE ROW LEVEL SECURITY;
ALTER TABLE sales.orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE sales.orders FORCE ROW LEVEL SECURITY;
ALTER TABLE sales.payments ENABLE ROW LEVEL SECURITY;
ALTER TABLE sales.payments FORCE ROW LEVEL SECURITY;

-- A user sees only their own access rows ...
CREATE POLICY region_access_self ON acl.region_access FOR SELECT
    USING (user_id = current_setting('app.user_id', true));

-- ... customers in those regions ...
CREATE POLICY customers_by_region ON sales.customers FOR SELECT
    USING (region IN (
        SELECT ra.region FROM acl.region_access AS ra
        WHERE ra.user_id = current_setting('app.user_id', true)
    ));

-- ... and orders/payments of the customers they can see (subqueries are RLS-filtered too).
CREATE POLICY orders_by_customer ON sales.orders FOR SELECT
    USING (customer_id IN (SELECT c.id FROM sales.customers AS c));

CREATE POLICY payments_by_order ON sales.payments FOR SELECT
    USING (order_id IN (SELECT o.id FROM sales.orders AS o));

RESET ROLE;

GRANT USAGE ON SCHEMA sales, acl TO acl_app;
GRANT SELECT ON sales.customers, sales.orders, sales.payments TO acl_app;
GRANT SELECT ON acl.region_access TO acl_app;
