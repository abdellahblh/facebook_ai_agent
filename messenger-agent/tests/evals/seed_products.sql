-- Seed catalog for the English test dataset. 12 products, ALL non-apparel
-- (electronics / home goods) -- deliberately no clothing, footwear, or
-- anything with a size attribute, per instruction. Ground truth for
-- sql_filter_specific and correctness_no_hallucination rows.
--
-- page_id is required and non-optional in the schema (app/db/models.py) --
-- using 'demo' as a single tenant here. Swap it to your real page_id.
--
-- price is a STRING column in the current schema (String(64)) -- see
-- BUGS_FOUND.md item 3 for why comparing it with <=/>= will not work as
-- written in repo.py today.

INSERT INTO products (id, page_id, name, price, stock, description) VALUES
  (gen_random_uuid(), 'demo', 'Wireless Earbuds Pro',        '4500 DA', 12, 'bluetooth 5.3, noise isolation, 24h battery case'),
  (gen_random_uuid(), 'demo', 'Bluetooth Speaker Mini',      '3200 DA', 0,  'portable, water resistant, 10h battery'),
  (gen_random_uuid(), 'demo', 'Stainless Steel Water Bottle', '900 DA', 40, '600ml, keeps cold 24h, keeps hot 12h'),
  (gen_random_uuid(), 'demo', 'Ceramic Coffee Mug Set',      '1500 DA', 25, 'set of 4, dishwasher safe, 350ml each'),
  (gen_random_uuid(), 'demo', 'LED Desk Lamp',               '2800 DA', 8,  'adjustable arm, 3 brightness levels, USB powered'),
  (gen_random_uuid(), 'demo', 'Portable Power Bank 20000mAh','3900 DA', 15, 'fast charge, dual USB output, digital display'),
  (gen_random_uuid(), 'demo', 'Wireless Mouse',              '1800 DA', 30, 'ergonomic, silent click, 2.4GHz'),
  (gen_random_uuid(), 'demo', 'Mechanical Keyboard',         '7200 DA', 5,  'RGB backlit, blue switches, USB-C'),
  (gen_random_uuid(), 'demo', 'Aluminum Phone Stand',        '700 DA',  50, 'adjustable angle, foldable, desk and bed use'),
  (gen_random_uuid(), 'demo', 'Noise Cancelling Headphones','12000 DA', 3,  'over-ear, 30h battery, active noise cancelling'),
  (gen_random_uuid(), 'demo', 'Smart Fitness Watch',         '8500 DA', 0,  'heart rate monitor, sleep tracking, waterproof'),
  (gen_random_uuid(), 'demo', 'Travel Backpack 30L',         '4200 DA', 18, 'laptop compartment, water resistant, USB charging port');

-- Ground truth for filter_* rows, computed from the values above:
--   cheapest:            Aluminum Phone Stand      700 DA   (filter_01)
--   most expensive:      Noise Cancelling Headphones 12000 DA (filter_02)
--   under 2000 DA:       Water Bottle(900), Mug Set(1500), Phone Stand(700), Mouse(1800)  (filter_03)
--   between 3000-5000:   Speaker(3200), Power Bank(3900), Backpack(4200), Earbuds(4500)   (filter_04)
--   under 5000 AND in stock: same as above MINUS Speaker (stock=0)                        (filter_05)
--   out of stock (stock=0): Bluetooth Speaker Mini, Smart Fitness Watch                   (filter_06, filter_07)
