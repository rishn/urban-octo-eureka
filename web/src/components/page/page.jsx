import React from "react";
import "./page.css";
import Card from "../cards/cards";
import PieChart from "../piechart/piechart";

function Page() {
  return (
    <div className="page">
      <Card />
      <Card />
      <PieChart />
    </div>
  );
}

export default Page;
