const fs = require('fs');
const csv = require('csv-parser');
const mysql = require('mysql2/promise');

async function insertCSVToDB() {
    const connection = await mysql.createConnection({
        host: 'localhost',
        user: 'root',
        password: '',
        database: 'resort_booking'
    });

    let rows = [];
    fs.createReadStream('synthetic_bookings_2024_2025.csv')
      .pipe(csv())
      .on('data', (data) => rows.push(data))
      .on('end', async () => {
          console.log(`Read ${rows.length} rows from CSV`);

          for (const row of rows) {
              await connection.query(
                  "INSERT INTO bookings_daily (check_in_date, total_bookings) VALUES (?, ?)",
                  [row.check_in_date, row.total_bookings]
              );
          }

          console.log(`Inserted ${rows.length} rows into MySQL`);
          await connection.end();
      });
}

insertCSVToDB();